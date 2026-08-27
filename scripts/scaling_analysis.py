"""
Scaling Analysis for TabPFN Multilabel Strategies.

Usage:
    python scripts/scaling_analysis.py --strategy br
    python scripts/scaling_analysis.py --strategy all
"""
import argparse
import datetime
import gc
import json
import pathlib
import sys
import time

import mlflow
import numpy as np
import pandas as pd
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS, load_dataset
from src.models import (
    BinaryRelevance,
    ClassifierChain,
    EnsembleClassifierChain,
    LabelPowerset,
    ProbabilisticClassifierChain,
)
from src.utils import compute_all_metrics, set_global_seeds
from src.utils.run_logger import RunLogger

# ── experiment config ─────────────────────────────────────────────────────────
SEEDS = [42, 1665, 8914]
BASE_SEED = 42

STRATEGIES = {
    "br":  (BinaryRelevance,              {"n_estimators": 16}),
    "lp":  (LabelPowerset,                {"n_estimators": 16}),
    "cc":  (ClassifierChain,              {"n_estimators": 16}),
    "ecc": (EnsembleClassifierChain,      {"n_estimators": 8, "n_chains": 8}),
    "pcc": (ProbabilisticClassifierChain, {"n_estimators": 16}),
}

DATA_DIR    = REPO_ROOT / "data" / "final"
RESULTS_DIR = REPO_ROOT / "results" / "scaling_benchmark"

def _free_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _vram_used_gb(device_idx: int = 0) -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_idx) / 1e9
    return 0.0

def _log_vram(label: str, device_idx: int = 0):
    used = _vram_used_gb(device_idx)
    if used > 0:
        print(f"  [VRAM] {label}: {used:.2f} GB allocated on GPU {device_idx}", flush=True)

def _predict_chunked(model, X: np.ndarray, chunk_size: int = 2000) -> np.ndarray:
    """
    Chunked predict_proba to bound VRAM.

    Note: chunking cannot rescue an ill-conditioned fit. TabPFN fits its
    preprocessing (including a low-rank SVD) on the TRAINING rows, which are
    identical no matter how the query rows are batched, so a torch _LinAlgError
    raised there recurs at every chunk size. Such failures are handled per cell
    in main() instead, by recording the cell as failed and continuing the sweep.
    """
    chunks = []
    for start in range(0, len(X), chunk_size):
        chunk = X[start : start + chunk_size]
        p = model.predict_proba(chunk)
        chunks.append(np.asarray(p, dtype=np.float32))
    return np.concatenate(chunks, axis=0)

def load_train_test():
    X1, Y1, feat_cols = load_dataset(str(DATA_DIR / "final_fold1.csv"))
    X2, Y2, _         = load_dataset(str(DATA_DIR / "final_fold2.csv"), feature_cols=feat_cols)
    X_te, Y_te, _     = load_dataset(str(DATA_DIR / "final_test.csv"),  feature_cols=feat_cols)
    X_train = np.concatenate([X1, X2], axis=0)
    Y_train = np.concatenate([Y1, Y2], axis=0)
    del X1, X2, Y1, Y2
    gc.collect()
    return X_train, Y_train, X_te, Y_te

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="br", choices=list(STRATEGIES.keys()) + ["all"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--chunk_size", type=int, default=20000)
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="Training-set sizes to sweep. Defaults to "
                             "250 1250 6250 31250 plus the full training set. "
                             "Pass an explicit list to skip the expensive large "
                             "sizes (PCC inference cost grows with context size).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory for the results JSON/CSV "
                             "(default: results/scaling_benchmark)")
    return parser.parse_args()

def main():
    args = parse_args()

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("tabamp_scaling_analysis")

    set_global_seeds(BASE_SEED)
    print(f"Global RNGs seeded with base seed {BASE_SEED}", flush=True)

    print("Loading data ...", flush=True)
    X_train_full, Y_train_full, X_test, Y_test = load_train_test()
    full_size = len(X_train_full)
    print(f"  Train: {X_train_full.shape}   Test: {X_test.shape}\n", flush=True)

    strats_to_run = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    # Stamped up front so incremental writes and the final write share one name.
    run_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    strat_suffix = args.strategy if args.strategy != "all" else "all_strats"
    out_json = out_dir / f"scaling_{strat_suffix}_{run_ts}.json"
    out_csv = out_dir / f"scaling_{strat_suffix}_{run_ts}.csv"

    def _checkpoint(results):
        """Persist after every cell so a late crash cannot discard earlier work."""
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        pd.DataFrame(results).to_csv(out_csv, index=False)

    # Target scaling sizes
    target_sizes = args.sizes if args.sizes else [250, 1250, 6250, 31250, full_size]
    print(f"  Sizes: {target_sizes}", flush=True)
    
    all_results = []
    
    for strat_name in strats_to_run:
        model_cls, model_kwargs = STRATEGIES[strat_name]
        print(f"\n{'='*60}")
        print(f"  Strategy: {strat_name.upper()}")
        print(f"{'='*60}", flush=True)

        for seed in SEEDS:
            print(f"\n  --- Seed: {seed} ---", flush=True)
            
            # Create a reproducible shuffle for this seed
            rng = np.random.RandomState(seed)
            shuffled_indices = rng.permutation(full_size)
            X_train_shuffled = X_train_full[shuffled_indices]
            Y_train_shuffled = Y_train_full[shuffled_indices]
            
            for size in target_sizes:
                print(f"    - Fitting on dataset size: {size}", flush=True)
                X_sub = X_train_shuffled[:size]
                Y_sub = Y_train_shuffled[:size]
                
                valid_label_indices = []
                for i in range(Y_sub.shape[1]):
                    if len(np.unique(Y_sub[:, i])) >= 2:
                        valid_label_indices.append(i)
                    else:
                        print(f"      Warning: Only one class present for label {LABEL_COLS[i]} at size {size}. Skipping this label.", flush=True)

                if len(valid_label_indices) == 0:
                    res_dict = {
                        "strategy": strat_name.upper(),
                        "seed": seed,
                        "dataset_size": size,
                        "train_time_s": 0.0,
                        "infer_time_s": 0.0,
                        "macro_ap": 0.0,
                        "macro_max_f1": 0.0,
                        "subset_accuracy": 0.0,
                        "macro_ece": 0.0,
                        "macro_brier": 0.0,
                    }
                    all_results.append(res_dict)
                    continue
                    
                Y_sub_valid = Y_sub[:, valid_label_indices]

                _log_vram("before fit")
                kwargs = {**model_kwargs, "seed": seed, "device": args.device}
                model = model_cls(**kwargs)

                try:
                    t_train = time.perf_counter()
                    model.fit(X_sub, Y_sub_valid)
                    train_time = time.perf_counter() - t_train
                    _log_vram("after fit")
                
                    print(f"      Train time: {train_time:.1f}s", flush=True)
                
                    print(f"      Predicting on test set (chunk={args.chunk_size})...", flush=True)
                    t_infer = time.perf_counter()
                    y_probs_test_valid = _predict_chunked(model, X_test, chunk_size=args.chunk_size)
                
                    # Stitch back into full prediction matrix with zeros for missing labels
                    y_probs_test = np.zeros((X_test.shape[0], len(LABEL_COLS)), dtype=np.float32)
                    for new_idx, orig_idx in enumerate(valid_label_indices):
                        y_probs_test[:, orig_idx] = y_probs_test_valid[:, new_idx]

                    infer_time = time.perf_counter() - t_infer
                    _log_vram("after inference")
                    print(f"      Infer time: {infer_time:.1f}s", flush=True)
                
                    _free_memory(model)
                
                    metrics = compute_all_metrics(
                        y_true=Y_test,
                        y_probs=y_probs_test,
                        label_names=LABEL_COLS,
                    )
                
                    print(f"      mAP={metrics['macro_ap']:.4f}  max_F1={metrics['macro_max_f1']:.4f}", flush=True)
                
                    res_dict = {
                        "strategy": strat_name.upper(),
                        "seed": seed,
                        "dataset_size": size,
                        "train_time_s": round(train_time, 2),
                        "infer_time_s": round(infer_time, 2),
                        "macro_ap": metrics["macro_ap"],
                        "macro_max_f1": metrics["macro_max_f1"],
                        "subset_accuracy": metrics["subset_accuracy"],
                        "macro_ece": metrics["calibration"]["macro_ece"],
                        "macro_brier": metrics["calibration"]["macro_brier"],
                    }
                
                    all_results.append(res_dict)
                    _checkpoint(all_results)
                    _free_memory()
                except Exception as exc:
                    # One degenerate cell must not discard the rest of the sweep.
                    # TabPFN's preprocessing SVD can fail to converge on certain
                    # training subsets; record it and move on.
                    print(f"      [FAILED] {type(exc).__name__}: {str(exc)[:160]}",
                          flush=True)
                    all_results.append({
                        "strategy": strat_name.upper(),
                        "seed": seed,
                        "dataset_size": size,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc)[:300],
                    })
                    _checkpoint(all_results)
                    _free_memory()

    # Final save (identical paths to the incremental checkpoints)
    _checkpoint(all_results)
    
    print(f"\nSaved scaling results to:\n  {out_json}\n  {out_csv}")

if __name__ == "__main__":
    with RunLogger(
        script_name="scaling_analysis.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=10.0,
        extra_meta={"argv": sys.argv},
    ):
        main()
