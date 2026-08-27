"""
Experiment 4: Joint-vs-product held-out likelihood.

Compares the mean held-out log-likelihood of the true test label vectors 
under the independent product of BR marginals against the PCC joint distribution.

Computed on the four real activities (Antimicrobial excluded).
"""
import datetime
import gc
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS, load_dataset
from src.models import ProbabilisticClassifierChain
from src.utils import set_global_seeds
from src.utils.run_logger import RunLogger

SEEDS = [42, 1665, 8914]
BASE_SEED = 42

ANALYSIS_LABELS = ['Antibacterial', 'Antifungal', 'Antiviral', 'Antiparasitic']

DATA_DIR    = REPO_ROOT / "data" / "final"
PREDS_DIR   = REPO_ROOT / "data" / "preds_prob"
RESULTS_DIR = REPO_ROOT / "results" / "label_dependence"

def _free_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_train_test():
    X1, Y1, feat_cols = load_dataset(str(DATA_DIR / "final_fold1.csv"))
    X2, Y2, _         = load_dataset(str(DATA_DIR / "final_fold2.csv"), feature_cols=feat_cols)
    X_te, Y_te, _     = load_dataset(str(DATA_DIR / "final_test.csv"),  feature_cols=feat_cols)
    
    X_train = np.concatenate([X1, X2], axis=0)
    Y_train = np.concatenate([Y1, Y2], axis=0)
    del X1, X2, Y1, Y2
    _free_memory()
    
    # Filter to ANALYSIS_LABELS
    col_indices = [LABEL_COLS.index(c) for c in ANALYSIS_LABELS]
    Y_train_4 = Y_train[:, col_indices]
    Y_test_4 = Y_te[:, col_indices]
    
    return X_train, Y_train_4, X_te, Y_test_4, col_indices

def get_br_test_probs(seed, col_indices):
    """Load saved BR test probabilities and filter to analysis labels."""
    npy_path = PREDS_DIR / "br" / f"br_seed{seed}.npy"
    if not npy_path.exists():
        npy_path = PREDS_DIR / f"br_seed{seed}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Could not find BR predictions for seed {seed} under {PREDS_DIR}. "
                "Run `python scripts/evaluate.py` first: it fits all five transforms "
                "for all three seeds and writes the cache. See docs/DATA.md section 3."
            )
            
    data = np.load(npy_path, allow_pickle=True).item()
    p_test = data['test_probs']
    # If the file saved probabilities for all 5 labels, we take only the 4 needed
    if p_test.shape[1] == len(LABEL_COLS):
        p_test = p_test[:, col_indices]
    return p_test

def compute_pcc_log_likelihood(model, X, Y_true, chunk_size=2000):
    """
    Extracts the joint log-likelihood of the true label vectors under the PCC
    joint distribution, computed via the chain rule in LOG-SPACE.

    Log-space multiplication prevents float64 underflow that occurs when
    multiplying 4+ small probabilities together in linear space.
    Prefix evaluations within each step are run sequentially to avoid
    concurrent GPU calls on the same classifier (thread-safety).
    """
    import itertools

    X = np.asarray(X, dtype=np.float32)
    Y_true = np.asarray(Y_true, dtype=np.int32)
    n_samples = X.shape[0]
    n_labels = model.n_labels_
    classifiers = model._chain.classifiers_

    # Prebuilt combo lookup: natural binary order matches itertools.product order
    all_combos = list(itertools.product([0, 1], repeat=n_labels))
    combo_to_idx = {combo: idx for idx, combo in enumerate(all_combos)}
    n_combos = len(all_combos)

    all_likelihoods = []

    for start in range(0, n_samples, chunk_size):
        X_chunk = X[start : start + chunk_size]
        Y_chunk = Y_true[start : start + chunk_size]
        chunk_len = X_chunk.shape[0]

        # ── Build memo: memo[step][prefix_tuple] = P(label_idx=1 | X, prefix) ──
        # Evaluated SEQUENTIALLY per step to guarantee GPU thread safety.
        memo: dict[int, dict] = {}
        for step in range(n_labels):
            memo[step] = {}
            prefixes = list(itertools.product([0, 1], repeat=step))
            for prefix_tuple in prefixes:
                if step == 0:
                    X_aug = X_chunk
                else:
                    aug_cols = np.tile(
                        np.array(prefix_tuple, dtype=np.float32), (chunk_len, 1)
                    )
                    X_aug = np.column_stack([X_chunk, aug_cols])
                proba = classifiers[step].predict_proba(X_aug)
                memo[step][prefix_tuple] = proba[:, 1]

        # ── Compute log joint probability for every combo in LOG-SPACE ──────────
        log_joint = np.zeros((chunk_len, n_combos), dtype=np.float64)

        for c_idx, combo in enumerate(all_combos):
            log_p = np.zeros(chunk_len, dtype=np.float64)
            for step, label_idx in enumerate(model.order_):
                prefix = tuple(combo[model.order_[i]] for i in range(step))
                bit = combo[label_idx]
                p1 = np.clip(memo[step][prefix], 1e-12, 1.0 - 1e-12)
                log_p += np.log(p1) if bit == 1 else np.log(1.0 - p1)
            log_joint[:, c_idx] = log_p

        # Normalise in log-space via log-sum-exp for numerical stability
        log_normaliser = np.logaddexp.reduce(log_joint, axis=1, keepdims=True)
        log_joint_norm = log_joint - log_normaliser      # shape (chunk_len, n_combos)

        # ── Extract log-probability of the TRUE label vector per sample ──────────
        chunk_indices = np.array(
            [combo_to_idx[tuple(row)] for row in Y_chunk.tolist()],
            dtype=np.int64,
        )
        chunk_ll = log_joint_norm[np.arange(chunk_len), chunk_indices]

        # Guard against any residual NaN (should not occur with log-space)
        n_nan = np.isnan(chunk_ll).sum()
        if n_nan > 0:
            print(f"  [WARNING] {n_nan}/{chunk_len} NaN log-likelihoods in chunk "
                  f"starting at {start}; replacing with -30 (log ~1e-13).", flush=True)
            chunk_ll = np.where(np.isnan(chunk_ll), -30.0, chunk_ll)

        all_likelihoods.append(chunk_ll)

    return np.concatenate(all_likelihoods)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seeds(BASE_SEED)
    print(f"Global RNGs seeded with base seed {BASE_SEED}", flush=True)

    print("Loading data (restricted to 4 analysis labels)...", flush=True)
    X_train, Y_train_4, X_test, Y_test_4, col_indices = load_train_test()
    print(f"  Train: X={X_train.shape}, Y={Y_train_4.shape}   Test: X={X_test.shape}, Y={Y_test_4.shape}\n", flush=True)

    all_seed_results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  Seed: {seed}")
        print(f"{'='*60}", flush=True)
        
        # 1. BR Log-Likelihood
        print(f"  -> Loading BR predictions for seed {seed}...", flush=True)
        p_br_4 = get_br_test_probs(seed, col_indices)
        p_br_4_clipped = np.clip(p_br_4, 1e-12, 1.0 - 1e-12)
        ll_br_marginal = Y_test_4 * np.log(p_br_4_clipped) + (1 - Y_test_4) * np.log(1 - p_br_4_clipped)
        ll_br = np.sum(ll_br_marginal, axis=1)
        mean_ll_br = np.mean(ll_br)
        print(f"     Mean BR Log-Likelihood: {mean_ll_br:.4f}", flush=True)
        
        # 2. Train PCC and get Joint Log-Likelihood
        print(f"  -> Fitting PCC model (4 labels) for seed {seed}...", flush=True)
        t_fit = time.perf_counter()
        pcc = ProbabilisticClassifierChain(device="cuda:0", n_estimators=16, seed=seed)
        pcc.fit(X_train, Y_train_4)
        print(f"     Fit time: {time.perf_counter() - t_fit:.1f}s", flush=True)
        
        print(f"  -> Extracting PCC joint log-likelihoods for Y_test...", flush=True)
        t_infer = time.perf_counter()
        ll_pcc = compute_pcc_log_likelihood(pcc, X_test, Y_test_4, chunk_size=2000)
        print(f"     Infer time: {time.perf_counter() - t_infer:.1f}s", flush=True)
        mean_ll_pcc = np.mean(ll_pcc)
        print(f"     Mean PCC Log-Likelihood: {mean_ll_pcc:.4f}", flush=True)
        
        # 3. Bootstrap CI of difference
        diffs = ll_pcc - ll_br
        n_test = len(diffs)
        rng_boot = np.random.RandomState(4242 + seed)
        
        mean_diffs = []
        for _ in range(2000):
            idx = rng_boot.choice(n_test, size=n_test, replace=True)
            mean_diffs.append(np.mean(diffs[idx]))
            
        mean_diffs = np.array(mean_diffs)
        ci_lower, ci_upper = np.percentile(mean_diffs, [2.5, 97.5])
        p_val = np.mean(mean_diffs <= 0)
        
        print(f"  -> Delta Log-Likelihood (PCC - BR): {np.mean(diffs):.4f}")
        print(f"     95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]  |  p-val: {p_val:.4f}")
        
        all_seed_results.append({
            "seed": seed,
            "mean_ll_br": mean_ll_br,
            "mean_ll_pcc": mean_ll_pcc,
            "delta_ll_mean": float(np.mean(diffs)),
            "delta_ll_ci_lower": float(ci_lower),
            "delta_ll_ci_upper": float(ci_upper),
            "pseudo_p_value": float(p_val),
            "significant_advantage": float(ci_lower) > 0
        })
        
        _free_memory(pcc)
        
    # Aggregate and Save
    run_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_json = RESULTS_DIR / f"e4_joint_likelihood_dependence_{run_ts}.json"
    with open(out_json, "w") as f:
        json.dump(all_seed_results, f, indent=2, default=float)
        
    print(f"\nSaved Experiment 4 results to: {out_json}")
    
    # Print overall summary across seeds
    df = pd.DataFrame(all_seed_results)
    print("\n--- Summary Across Seeds ---")
    print(df[["seed", "mean_ll_br", "mean_ll_pcc", "delta_ll_mean", "significant_advantage"]].to_string(index=False))


if __name__ == "__main__":
    with RunLogger(
        script_name="joint_likelihood_dependence.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=10.0,
        extra_meta={"argv": sys.argv},
    ):
        main()
