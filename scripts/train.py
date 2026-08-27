"""
Train a single multilabel strategy for one seed and log results to MLflow.

Data layout:
    data/final/final_fold1.csv  — training fold 1
    data/final/final_fold2.csv  — training fold 2  (concatenated for full training set)
    data/final/final_test.csv   — held-out test set

Honest threshold selection:
    Thresholds τ*_j are selected on 5-fold OOF predictions on the training set,
    then frozen before the test set is evaluated (no leakage).

Usage:
    python scripts/train.py --strategy br --seed 42
    python scripts/train.py --strategy ecc --seed 1665 --n_estimators 8 --n_chains 8
"""
import argparse
import json
import pathlib
import sys
import time

from src.utils.run_logger import RunLogger

import mlflow
import numpy as np

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
from src.utils import compute_all_metrics, generate_oof_probs, set_global_seeds

# ── strategy registry ─────────────────────────────────────────────────────────
STRATEGY_MAP = {
    "br":  BinaryRelevance,
    "lp":  LabelPowerset,
    "cc":  ClassifierChain,
    "ecc": EnsembleClassifierChain,
    "pcc": ProbabilisticClassifierChain,
}

# Default n_estimators per strategy (from info.tex)
N_ESTIMATORS_DEFAULT = {"br": 16, "lp": 16, "cc": 16, "ecc": 8, "pcc": 16}

DATA_DIR    = REPO_ROOT / "data" / "final"
RESULTS_DIR = REPO_ROOT / "results"


def load_train_test():
    """Concatenate fold1 + fold2 for training; load test split separately."""
    X1, Y1, feat_cols = load_dataset(str(DATA_DIR / "final_fold1.csv"))
    X2, Y2, _         = load_dataset(str(DATA_DIR / "final_fold2.csv"), feature_cols=feat_cols)
    X_te, Y_te, _     = load_dataset(str(DATA_DIR / "final_test.csv"),  feature_cols=feat_cols)
    return np.concatenate([X1, X2]), np.concatenate([Y1, Y2]), X_te, Y_te


def parse_args():
    parser = argparse.ArgumentParser(description="Train one TabPFN multilabel strategy")
    parser.add_argument("--strategy",     type=str, default="br",
                        choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=None,
                        help="TabPFN ensemble size (default per strategy from info.tex)")
    parser.add_argument("--n_chains",     type=int, default=8,
                        help="Number of chains for ECC")
    parser.add_argument("--device",       type=str, default="auto")
    parser.add_argument("--oof_splits",   type=int, default=5,
                        help="Number of CV folds for OOF threshold selection")
    return parser.parse_args()


def main():
    args = parse_args()
    n_est = args.n_estimators or N_ESTIMATORS_DEFAULT[args.strategy]

    # Seed all global RNGs before any data loading or model construction.
    # This ensures the run is exactly reproducible from --seed alone.
    set_global_seeds(args.seed)
    print(f"Global RNGs seeded with seed={args.seed}", flush=True)

    # ── model kwargs shared across OOF folds and the final model ─────────────
    model_kwargs = dict(device=args.device, n_estimators=n_est, seed=args.seed)
    if args.strategy == "ecc":
        model_kwargs["n_chains"] = args.n_chains

    def model_factory():
        return STRATEGY_MAP[args.strategy](**model_kwargs)

    # Pin the tracking store, as evaluate.py and scaling_analysis.py do. MLflow 3
    # refuses the default ./mlruns file store outright, so without this the run
    # dies before any modelling happens.
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("tabamp_multilabel_classification")

    with mlflow.start_run(run_name=f"{args.strategy}_seed{args.seed}"):
        # ── data ─────────────────────────────────────────────────────────────
        print("Loading data (fold1 + fold2 → train, final_test → test) ...")
        X_train, Y_train, X_test, Y_test = load_train_test()
        print(f"  Train: {X_train.shape}   Test: {X_test.shape}")

        # ── log params ───────────────────────────────────────────────────────
        mlflow.log_params({
            "strategy":       args.strategy,
            "seed":           args.seed,
            "n_estimators":   n_est,
            "device":         args.device,
            "oof_splits":     args.oof_splits,
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "n_features":     X_train.shape[1],
            "tabpfn_version": __import__("tabpfn").__version__,
        })
        if args.strategy == "ecc":
            mlflow.log_param("n_chains", args.n_chains)

        # ── OOF predictions for honest threshold selection ────────────────────
        # Each sample's threshold-selection probability comes from a model
        # that never saw that sample — eliminating in-sample optimism.
        print(f"\nGenerating {args.oof_splits}-fold OOF predictions for threshold selection ...")
        t_oof = time.perf_counter()
        oof_probs = generate_oof_probs(
            model_factory=model_factory,
            X=X_train, Y=Y_train,
            n_splits=args.oof_splits,
            seed=args.seed,
        )
        oof_time = time.perf_counter() - t_oof
        print(f"  OOF time: {oof_time:.1f}s")
        mlflow.log_metric("oof_time_s", round(oof_time, 2))

        # ── final model trained on ALL training data ──────────────────────────
        print(f"\nTraining final '{args.strategy}' on full training set ...")
        t_train = time.perf_counter()
        model = model_factory()
        model.fit(X_train, Y_train)
        train_time = time.perf_counter() - t_train
        print(f"  Training time: {train_time:.1f}s")
        mlflow.log_metric("train_time_s", round(train_time, 2))

        # ── inference on test set ─────────────────────────────────────────────
        t_infer = time.perf_counter()
        y_probs_test = model.predict_proba(X_test)
        infer_time = time.perf_counter() - t_infer
        print(f"  Inference time: {infer_time:.1f}s")
        mlflow.log_metric("infer_time_s", round(infer_time, 2))

        # ── evaluate — honest thresholds from OOF, applied to test ───────────
        metrics = compute_all_metrics(
            y_true=Y_test,
            y_probs=y_probs_test,
            label_names=LABEL_COLS,
            y_true_train=Y_train,
            y_probs_train=oof_probs,   # ← OOF, not in-sample
        )

        # ── log scalar metrics ────────────────────────────────────────────────
        mlflow.log_metrics({
            "macro_ap":         metrics["macro_ap"],
            "macro_max_f1":     metrics["macro_max_f1"],
            "macro_honest_f1":  metrics.get("macro_honest_f1", float("nan")),
            "subset_accuracy":  metrics["subset_accuracy"],
            "macro_ece":        metrics["calibration"]["macro_ece"],
            "macro_brier":      metrics["calibration"]["macro_brier"],
        })

        # ── save JSON artifact ────────────────────────────────────────────────
        RESULTS_DIR.mkdir(exist_ok=True)
        out_file = RESULTS_DIR / f"{args.strategy}_seed{args.seed}.json"
        import platform
        from src.utils.seed_utils import get_rng_states
        payload = {
            "strategy":        args.strategy,
            "seed":            args.seed,
            "n_estimators":    n_est,
            "oof_splits":      args.oof_splits,
            "device":          args.device,
            "tabpfn_version":  __import__("tabpfn").__version__,
            "python_version":  platform.python_version(),
            "rng_states":      get_rng_states(),
            "oof_time_s":      round(oof_time, 2),
            "train_time_s":    round(train_time, 2),
            "infer_time_s":    round(infer_time, 2),
            "metrics":         metrics,
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        mlflow.log_artifact(str(out_file))
        print(f"\nSaved results → {out_file}")

        # ── print summary ─────────────────────────────────────────────────────
        print("\n── Evaluation Summary ─────────────────────────────────────")
        print(f"  macro_ap         : {metrics['macro_ap']:.4f}")
        print(f"  macro_honest_f1  : {metrics.get('macro_honest_f1', float('nan')):.4f}  ← OOF thresholds")
        print(f"  macro_max_f1     : {metrics['macro_max_f1']:.4f}  ← test-tuned (optimistic bound)")
        print(f"  gap (optimism)   : {metrics['macro_max_f1'] - metrics.get('macro_honest_f1', 0):.4f}")
        print(f"  subset_accuracy  : {metrics['subset_accuracy']:.4f}")
        print(f"  macro_ece        : {metrics['calibration']['macro_ece']:.4f}")
        print(f"  macro_brier      : {metrics['calibration']['macro_brier']:.4f}")
        print(f"  oof_time_s       : {oof_time:.1f}s")
        print(f"  train_time_s     : {train_time:.1f}s")


if __name__ == "__main__":
    with RunLogger(
        script_name="train.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=10.0,
        extra_meta={"argv": sys.argv},
    ):
        main()
