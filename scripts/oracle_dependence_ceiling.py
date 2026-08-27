"""
Experiment 2: Oracle conditional-predictability ceiling.

The Oracle ceiling is a fixed theoretical upper bound: for each activity,
the model is given the TRUE sibling-label values as extra features. This
represents the maximum AP achievable by any joint-label method.

We compute the ceiling ONCE (per seed) and then compare it against the
saved test predictions of ALL five strategies (BR, CC, ECC, LP, PCC),
quantifying how much headroom each strategy leaves.

Usage:
    python scripts/oracle_dependence_ceiling.py [--device cuda:0]
"""
import argparse
import datetime
import gc
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS, load_dataset
from src.models.base import _get_tabpfn
from src.utils import set_global_seeds
from src.utils.run_logger import RunLogger

# ── Constants ──────────────────────────────────────────────────────────────────
SEEDS = [42, 1665, 8914]
BASE_SEED = 42

# Antimicrobial excluded — it is a logical OR of the other 4 and would inflate
# dependence metrics trivially.
ANALYSIS_LABELS = ["Antibacterial", "Antifungal", "Antiviral", "Antiparasitic"]
STRATEGIES      = ["br", "lp","cc","ecc","pcc"]

DATA_DIR         = REPO_ROOT / "data" / "final"
PREDS_DIR        = REPO_ROOT / "data" / "preds_prob"
ORACLE_PREDS_DIR = PREDS_DIR / "oracle"
RESULTS_DIR      = REPO_ROOT / "results" / "label_dependence"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _free_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_train_test():
    """Load full train + test splits restricted to the 4 analysis labels."""
    X1, Y1, feat_cols = load_dataset(str(DATA_DIR / "final_fold1.csv"))
    X2, Y2, _         = load_dataset(str(DATA_DIR / "final_fold2.csv"), feature_cols=feat_cols)
    X_te, Y_te, _     = load_dataset(str(DATA_DIR / "final_test.csv"),  feature_cols=feat_cols)

    X_train = np.concatenate([X1, X2], axis=0)
    Y_train = np.concatenate([Y1, Y2], axis=0)
    _free_memory(X1, X2, Y1, Y2)

    col_indices  = [LABEL_COLS.index(c) for c in ANALYSIS_LABELS]
    Y_train_4    = Y_train[:, col_indices]
    Y_test_4     = Y_te[:, col_indices]

    return X_train, Y_train_4, X_te, Y_test_4, col_indices


def load_strategy_test_probs(strategy: str, seed: int, col_indices: list) -> np.ndarray:
    """Load saved test probabilities for a strategy and restrict to analysis labels."""
    path = PREDS_DIR / strategy / f"{strategy}_seed{seed}.npy"
    data = np.load(path, allow_pickle=True).item()
    p    = data["test_probs"]                        # (N_test, 5)
    return p[:, col_indices]                          # (N_test, 4)


def _predict_chunked(model, X: np.ndarray, chunk_size: int = 20_000) -> np.ndarray:
    """Run predict_proba in chunks to bound VRAM usage."""
    chunks = []
    for start in range(0, len(X), chunk_size):
        p = model.predict_proba(X[start : start + chunk_size])[:, 1]
        chunks.append(np.asarray(p, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def compute_oracle_probs(
    X_train: np.ndarray,
    Y_train_4: np.ndarray,
    X_test: np.ndarray,
    Y_test_4: np.ndarray,
    seed: int,
    device: str,
) -> np.ndarray:
    """
    For each label j, augment X with the 3 TRUE sibling labels and train a
    TabPFN classifier. Returns P_oracle of shape (N_test, 4).
    """
    N_test   = X_test.shape[0]
    P_oracle = np.zeros((N_test, len(ANALYSIS_LABELS)), dtype=np.float32)

    for j, label_name in enumerate(ANALYSIS_LABELS):
        print(f"    Oracle fit [{label_name}] ...", flush=True)

        other_idx      = [i for i in range(len(ANALYSIS_LABELS)) if i != j]
        X_train_aug    = np.column_stack([X_train,  Y_train_4[:, other_idx].astype(np.float32)])
        X_test_aug     = np.column_stack([X_test,   Y_test_4[:, other_idx].astype(np.float32)])
        y_train_j      = Y_train_4[:, j].astype(int)

        clf = _get_tabpfn(device=device, n_estimators=16, seed=seed,
                          ignore_pretraining_limits=True)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_aug, y_train_j)
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_j = _predict_chunked(clf, X_test_aug, chunk_size=20_000)
        t_infer = time.perf_counter() - t0

        P_oracle[:, j] = p_j
        print(f"      fit={t_fit:.1f}s  infer={t_infer:.1f}s", flush=True)
        _free_memory(clf, X_train_aug, X_test_aug)

    return P_oracle


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Oracle ceiling for all strategies")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for TabPFN (e.g. 'cuda:0', 'cuda:1', 'auto')")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seeds(BASE_SEED)
    print(f"Global RNG seeded with base seed {BASE_SEED}", flush=True)

    print("Loading data (restricted to 4 analysis labels)...", flush=True)
    X_train, Y_train_4, X_test, Y_test_4, col_indices = load_train_test()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}\n", flush=True)

    all_rows = []

    for seed in SEEDS:
        print(f"\n{'='*60}\n  Seed: {seed}\n{'='*60}", flush=True)

        # ── Step 1: Oracle probabilities (computed once per seed) ──────────────
        print("  Computing Oracle probabilities ...", flush=True)
        P_oracle = compute_oracle_probs(
            X_train, Y_train_4, X_test, Y_test_4,
            seed=seed, device=args.device,
        )

        # Save Oracle test probabilities for reproducibility / later analysis
        oracle_npy = ORACLE_PREDS_DIR / f"oracle_seed{seed}.npy"
        np.save(oracle_npy, {
            "strategy":    "oracle",
            "seed":        seed,
            "label_cols":  ANALYSIS_LABELS,
            "test_probs":  P_oracle,   # shape (N_test, 4)
        })
        print(f"  Saved Oracle probs → {oracle_npy}", flush=True)

        # ── Step 2: Per-strategy comparison ───────────────────────────────────
        for strategy in STRATEGIES:
            print(f"  Loading strategy '{strategy}' predictions...", flush=True)
            P_strat = load_strategy_test_probs(strategy, seed, col_indices)

            for j, label_name in enumerate(ANALYSIS_LABELS):
                y_true  = Y_test_4[:, j]
                ap_base = average_precision_score(y_true, P_strat[:, j])
                ap_ora  = average_precision_score(y_true, P_oracle[:, j])
                lift    = ap_ora - ap_base

                all_rows.append({
                    "seed":         seed,
                    "strategy":     strategy.upper(),
                    "label":        label_name,
                    "AP_strategy":  float(ap_base),
                    "AP_oracle":    float(ap_ora),
                    "AP_lift":      float(lift),
                })
                print(f"    [{strategy.upper():3s}] {label_name:<15s}  "
                      f"AP_strategy={ap_base:.4f}  AP_oracle={ap_ora:.4f}  lift=+{lift:.4f}",
                      flush=True)

        _free_memory(P_oracle)

    # ── Save ───────────────────────────────────────────────────────────────────
    run_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_json = RESULTS_DIR / f"e2_oracle_ceiling_{run_ts}.json"
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2, default=float)
    print(f"\nSaved results → {out_json}")

    # ── Summary table ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    print("\n--- Mean across seeds: AP_lift (Oracle - Strategy) per label ---")
    pivot = (df.groupby(["strategy", "label"])["AP_lift"]
               .mean()
               .unstack("label")
               .reindex(columns=ANALYSIS_LABELS)
               .rename(index=str.upper))
    print(pivot.round(4).to_string())

    print("\n--- Macro-mAP and Macro Oracle lift per strategy ---")
    macro = df.groupby(["strategy", "seed"])[["AP_strategy", "AP_oracle"]].mean()
    macro_summary = macro.groupby("strategy")[["AP_strategy", "AP_oracle"]].mean()
    macro_summary["macro_lift"] = macro_summary["AP_oracle"] - macro_summary["AP_strategy"]
    print(macro_summary.round(4).to_string())


if __name__ == "__main__":
    with RunLogger(
        script_name="oracle_dependence_ceiling.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=10.0,
        extra_meta={"argv": sys.argv},
    ):
        main()
