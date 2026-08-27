"""
GPU and memory-optimised full benchmark: 5 strategies × 3 seeds.

Memory discipline:
  - Each (strategy, seed) run is self-contained: OOF models, final model, and
    intermediate tensors are explicitly deleted and VRAM is flushed between
    every phase (OOF fold → final train → inference).
  - Predictions are stored as float32 numpy arrays (never as GPU tensors).
  - Pooled bootstrap arrays are freed immediately after the CI is computed.
  - GPU memory is reported before and after each strategy block.

Honest threshold selection:
  Thresholds τ*_j are selected on 5-fold OOF training-set predictions,
  then frozen before the test set is evaluated (no leakage).

Produces:
  results/benchmark_all_seeds.json   — raw per-seed metrics
  results/benchmark_summary.csv      — aggregated table with CIs and Δ vs BR

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --device cuda:0 --oof_splits 5 --chunk_size 2000
"""
import argparse
import datetime
import gc
import hashlib
import json
import pathlib
import sys
import time

from src.utils.run_logger import RunLogger

import mlflow
import numpy as np
import pandas as pd

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
from src.utils import (
    compute_all_metrics,
    compute_ap,
    compute_bootstrap_ci,
    compute_paired_statistical_tests,
    generate_oof_probs,
    set_global_seeds,
)

# ── experiment config ─────────────────────────────────────────────────────────
SEEDS = [42, 1665, 8914]

STRATEGIES = {
    "BR":  (BinaryRelevance,              {"n_estimators": 16}),
    "LP":  (LabelPowerset,                {"n_estimators": 16}),
    "CC":  (ClassifierChain,              {"n_estimators": 16}),
    "ECC": (EnsembleClassifierChain,      {"n_estimators": 8, "n_chains": 8}),
    "PCC": (ProbabilisticClassifierChain, {"n_estimators": 16}),
}

DATA_DIR    = REPO_ROOT / "data" / "final"
RESULTS_DIR = REPO_ROOT / "results"
PREDS_DIR   = REPO_ROOT / "results" / "preds"


# ── GPU / memory helpers ──────────────────────────────────────────────────────
def _free_memory(*objs):
    """Delete objects, run GC, and flush the CUDA allocator cache."""
    for obj in objs:
        del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _vram_used_gb(device_idx: int = 0) -> float:
    """Return current VRAM used on a given GPU in GB, or 0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device_idx) / 1e9
    except Exception:
        pass
    return 0.0


def _log_vram(label: str, device_idx: int = 0):
    used = _vram_used_gb(device_idx)
    if used > 0:
        print(f"  [VRAM] {label}: {used:.2f} GB allocated on GPU {device_idx}", flush=True)


def _predict_chunked(model, X: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Run predict_proba in chunks to bound VRAM usage on large arrays.
    Results are collected as float32 numpy and concatenated on CPU.
    """
    chunks = []
    for start in range(0, len(X), chunk_size):
        chunk = X[start : start + chunk_size]
        p = model.predict_proba(chunk)
        chunks.append(np.asarray(p, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


# ── prediction persistence ───────────────────────────────────────────────────
def _probs_hash(*arrays: np.ndarray) -> str:
    """SHA-256 fingerprint of concatenated float32 probability arrays."""
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def _save_probs_npy(
    strat_name: str,
    seed: int,
    oof_probs: np.ndarray,
    y_probs_test: np.ndarray,
    label_cols: list,
    out_dir: pathlib.Path,
):
    """
    Save a structured .npy file (object array, allow_pickle=True) for one
    (strategy, seed) run.

    Saved keys
    ----------
    strategy         : str
    seed             : int
    label_cols       : list[str]   — ordered label names (5 AMP activities)
    sha256           : str         — SHA-256 over oof_probs || y_probs_test
    train_oof_probs  : float32 ndarray, shape (n_train, n_labels)
    test_probs       : float32 ndarray, shape (n_test,  n_labels)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _probs_hash(oof_probs, y_probs_test)
    payload = {
        "strategy":        strat_name,
        "seed":            seed,
        "label_cols":      label_cols,
        "sha256":          fingerprint,
        "train_oof_probs": np.asarray(oof_probs,    dtype=np.float32),
        "test_probs":      np.asarray(y_probs_test, dtype=np.float32),
    }
    out_path = out_dir / f"{strat_name.lower()}_seed{seed}.npy"
    np.save(str(out_path), payload)
    print(
        f"  [probs] Saved {strat_name}/seed={seed} → {out_path}  "
        f"sha256={fingerprint[:12]}...",
        flush=True,
    )


# ── data loading ──────────────────────────────────────────────────────────────
def load_train_test():
    """Concatenate fold1 + fold2 as training set; keep test split separate."""
    X1, Y1, feat_cols = load_dataset(str(DATA_DIR / "final_fold1.csv"))
    X2, Y2, _         = load_dataset(str(DATA_DIR / "final_fold2.csv"), feature_cols=feat_cols)
    X_te, Y_te, _     = load_dataset(str(DATA_DIR / "final_test.csv"),  feature_cols=feat_cols)
    X_train = np.concatenate([X1, X2], axis=0)
    Y_train = np.concatenate([Y1, Y2], axis=0)
    # Free intermediate fold arrays immediately
    del X1, X2, Y1, Y2
    gc.collect()
    return X_train, Y_train, X_te, Y_te


# ── single (strategy, seed) run ───────────────────────────────────────────────
def run_one(strat_name, model_cls, model_kwargs, seed, device, oof_splits, chunk_size,
            X_train, Y_train, X_test, Y_test):
    """
    Three-phase execution with explicit VRAM cleanup between phases:

    Phase 1 — OOF:    k fold models trained sequentially; each deleted after
                      its validation split is predicted.
    Phase 2 — Train:  Final model trained on the full training set.
    Phase 3 — Infer:  Chunked prediction on the test set; model deleted
                      immediately after probabilities are extracted.

    Honest thresholds are selected from OOF probs (Phase 1), never from
    in-sample or test predictions.
    """
    kwargs = {**model_kwargs, "seed": seed, "device": device}

    def model_factory():
        return model_cls(**kwargs)

    # ── Phase 1: OOF predictions ──────────────────────────────────────────────
    _log_vram("before OOF")
    print(f"  [seed={seed}] Phase 1 — {oof_splits}-fold OOF probs ...", flush=True)
    t_oof = time.perf_counter()
    oof_probs = generate_oof_probs(
        model_factory=model_factory,
        X=X_train, Y=Y_train,
        n_splits=oof_splits,
        seed=seed,
    )
    oof_time = time.perf_counter() - t_oof
    _free_memory()          # flush anything left after the last fold cleanup
    _log_vram("after OOF")

    # ── Phase 2: Final model on full training set ─────────────────────────────
    print(f"  [seed={seed}] Phase 2 — fitting final model ...", flush=True)
    t_train = time.perf_counter()
    model = model_factory()
    model.fit(X_train, Y_train)
    train_time = time.perf_counter() - t_train
    _log_vram("after final fit")

    # ── Phase 3: Chunked inference on test set ────────────────────────────────
    print(f"  [seed={seed}] Phase 3 — predicting on test set (chunk={chunk_size}) ...", flush=True)
    t_infer = time.perf_counter()
    y_probs_test = _predict_chunked(model, X_test, chunk_size=chunk_size)
    infer_time = time.perf_counter() - t_infer

    # Free model ASAP — probs are now numpy on CPU
    _free_memory(model)
    _log_vram("after inference + model free")

    # ── Evaluate — OOF thresholds, test predictions ───────────────────────────
    metrics = compute_all_metrics(
        y_true=Y_test,
        y_probs=y_probs_test,
        label_names=LABEL_COLS,
        y_true_train=Y_train,
        y_probs_train=oof_probs,   # ← honest: OOF, not in-sample
    )
    # NOTE: oof_probs is returned to the caller for .npy persistence;
    #       the caller is responsible for freeing it after saving.

    metrics["oof_time_s"]   = round(oof_time, 2)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["infer_time_s"] = round(infer_time, 2)

    honest_f1 = metrics.get("macro_honest_f1", float("nan"))
    gap = metrics["macro_max_f1"] - honest_f1
    print(
        f"  [seed={seed}] mAP={metrics['macro_ap']:.4f}  "
        f"honest_F1={honest_f1:.4f}  max_F1={metrics['macro_max_f1']:.4f}  "
        f"gap={gap:.4f}  oof={oof_time:.0f}s  train={train_time:.0f}s  infer={infer_time:.0f}s",
        flush=True,
    )
    return metrics, y_probs_test, oof_probs


# ── main benchmark loop ────────────────────────────────────────────────────────
def run_benchmark(device: str = "auto", oof_splits: int = 5, chunk_size: int = 2000):
    RESULTS_DIR.mkdir(exist_ok=True)
    # Explicitly point to the project SQLite backend so runs are always logged
    # to mlflow.db regardless of what MLFLOW_TRACKING_URI is set in the shell.
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("tabamp_multilabel_benchmark")

    # Seed all global RNGs (Python random, NumPy, PyTorch) for reproducibility.
    # Per-strategy seeds are derived as  SEEDS[i] + offset  inside each run.
    BASE_SEED = SEEDS[0]  # 42 — anchors the global state before data loading
    set_global_seeds(BASE_SEED)
    print(f"Global RNGs seeded with base seed {BASE_SEED}", flush=True)

    # One timestamped subfolder for all .npy files from this run
    run_ts   = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    preds_run_dir = PREDS_DIR / f"evaluate_{run_ts}"
    print(f"Predictions will be saved to: {preds_run_dir}", flush=True)

    print("Loading data ...", flush=True)
    X_train, Y_train, X_test, Y_test = load_train_test()
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}\n", flush=True)

    all_results = {}   # strategy → {seed → metrics}
    # Store only (Y_test, y_probs_test) per seed; Y_test is shared so no extra copy
    all_probs   = {}   # strategy → [(Y_test, y_probs_test), ...]

    for strat_name, (model_cls, model_kwargs) in STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"  Strategy: {strat_name}")
        print(f"{'='*60}", flush=True)
        _log_vram(f"start of {strat_name}")

        seed_metrics   = {}
        probs_per_seed = []   # holds (Y_test ref, probs array) for 3 seeds

        for seed in SEEDS:
            with mlflow.start_run(run_name=f"{strat_name}_seed{seed}", nested=False):
                mlflow.log_params({
                    "strategy":       strat_name,
                    "seed":           seed,
                    "device":         device,
                    "oof_splits":     oof_splits,
                    "chunk_size":     chunk_size,
                    "tabpfn_version": __import__("tabpfn").__version__,
                    **model_kwargs,
                })

                metrics, y_probs_test, oof_probs = run_one(
                    strat_name, model_cls, model_kwargs, seed, device,
                    oof_splits, chunk_size,
                    X_train, Y_train, X_test, Y_test,
                )

                mlflow.log_metrics({
                    "macro_ap":         metrics["macro_ap"],
                    "macro_max_f1":     metrics["macro_max_f1"],
                    "macro_honest_f1":  metrics.get("macro_honest_f1", float("nan")),
                    "subset_accuracy":  metrics["subset_accuracy"],
                    "macro_ece":        metrics["calibration"]["macro_ece"],
                    "macro_brier":      metrics["calibration"]["macro_brier"],
                    "oof_time_s":       metrics["oof_time_s"],
                    "train_time_s":     metrics["train_time_s"],
                    "infer_time_s":     metrics["infer_time_s"],
                })

                # ── persist prediction probabilities for this (strategy, seed) ──
                _save_probs_npy(
                    strat_name=strat_name,
                    seed=seed,
                    oof_probs=oof_probs,
                    y_probs_test=y_probs_test,
                    label_cols=LABEL_COLS,
                    out_dir=preds_run_dir,
                )
                # Free oof_probs now that it has been saved
                _free_memory(oof_probs)

                seed_metrics[seed] = metrics
                probs_per_seed.append((Y_test, y_probs_test))

        all_results[strat_name] = seed_metrics
        all_probs[strat_name]   = probs_per_seed

        # Flush GPU after every strategy block
        _free_memory()
        _log_vram(f"end of {strat_name}")

    # ── save raw results ──────────────────────────────────────────────────────
    import platform
    from src.utils.seed_utils import get_rng_states
    raw_path = RESULTS_DIR / "benchmark_all_seeds.json"
    raw_payload = {
        "run_metadata": {
            "seeds":           SEEDS,
            "base_seed":       SEEDS[0],
            "rng_states":      get_rng_states(),
            "device":          device,
            "oof_splits":      oof_splits,
            "chunk_size":      chunk_size,
            "tabpfn_version":  __import__("tabpfn").__version__,
            "python_version":  platform.python_version(),
        },
        "results": all_results,
    }
    with open(raw_path, "w") as f:
        json.dump(raw_payload, f, indent=2, default=float)
    print(f"\nSaved raw results → {raw_path}")


    # ── aggregate and build summary table ─────────────────────────────────────
    summary_rows = []
    br_aps = [all_results["BR"][s]["macro_ap"] for s in SEEDS]

    for strat_name in STRATEGIES:
        s_res = all_results[strat_name]

        macro_aps  = [s_res[s]["macro_ap"]                          for s in SEEDS]
        honest_f1s = [s_res[s].get("macro_honest_f1", float("nan")) for s in SEEDS]
        max_f1s    = [s_res[s]["macro_max_f1"]                      for s in SEEDS]
        sub_accs   = [s_res[s]["subset_accuracy"]                   for s in SEEDS]
        eces       = [s_res[s]["calibration"]["macro_ece"]          for s in SEEDS]
        briers     = [s_res[s]["calibration"]["macro_brier"]        for s in SEEDS]
        train_t    = [s_res[s]["train_time_s"]                      for s in SEEDS]
        infer_t    = [s_res[s]["infer_time_s"]                      for s in SEEDS]
        gaps       = [mf - hf for mf, hf in zip(max_f1s, honest_f1s) if not np.isnan(hf)]

        # 95% bootstrap CI — pool predictions across seeds, free pooled arrays after
        Y_pool = np.vstack([p[0] for p in all_probs[strat_name]])
        P_pool = np.vstack([p[1] for p in all_probs[strat_name]])
        _, ci_lo, ci_hi = compute_bootstrap_ci(
            Y_pool, P_pool, metric_fn=compute_ap, n_bootstraps=2000
        )
        del Y_pool, P_pool   # free pooled arrays immediately after CI
        gc.collect()

        paired = compute_paired_statistical_tests(br_aps, macro_aps)

        summary_rows.append({
            "Strategy":            strat_name,
            "mAP (mean±std)":      f"{np.mean(macro_aps):.4f} ± {np.std(macro_aps):.4f}",
            "mAP 95% CI":          f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            "Honest-F1 (mean)":    f"{np.nanmean(honest_f1s):.4f}",
            "Max-F1 (mean)":       f"{np.mean(max_f1s):.4f}",
            "Gap (Max-Honest F1)": f"{np.mean(gaps):.4f}" if gaps else "N/A",
            "Subset Acc (mean)":   f"{np.mean(sub_accs):.4f}",
            "ECE (mean)":          f"{np.mean(eces):.4f}",
            "Brier (mean)":        f"{np.mean(briers):.4f}",
            "Δ mAP vs BR":         f"{paired['delta_mean']:+.4f}",
            "Wilcoxon p":          f"{paired['wilcoxon_p']:.3f}" if not np.isnan(paired["wilcoxon_p"]) else "N/A",
            "Train time (mean)":   f"{np.mean(train_t):.0f}s",
            "Infer time (mean)":   f"{np.mean(infer_t):.0f}s",
        })

    # Free all stored predictions now that CI computation is done
    del all_probs
    gc.collect()

    df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "="*80)
    print("  TABPFN MULTILABEL BENCHMARK — 5 strategies × 3 seeds")
    print(f"  Honest thresholds: {oof_splits}-fold OOF  |  Inference chunk: {chunk_size}")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nSaved summary → {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="GPU-optimised multilabel benchmark")
    parser.add_argument("--device",     type=str, default="auto",
                        help="TabPFN device  (auto | cuda:0 | cuda:1 | cpu)")
    parser.add_argument("--oof_splits", type=int, default=5,
                        help="CV folds for OOF threshold selection (default 5)")
    parser.add_argument("--chunk_size", type=int, default=2000,
                        help="Test-set prediction chunk size to bound VRAM (default 2000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with RunLogger(
        script_name="evaluate.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=10.0,
        extra_meta={"argv": sys.argv, "device": args.device},
    ):
        run_benchmark(device=args.device, oof_splits=args.oof_splits, chunk_size=args.chunk_size)
