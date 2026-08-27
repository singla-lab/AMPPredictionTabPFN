"""
Classifier Chain context ablation: soft vs hard label context at inference.

ClassifierChain.fit conditions each step on the *ground-truth* binary labels of
the preceding steps, so the context columns seen during training are exactly
{0, 1}. At inference the default (probabilistic=True) instead propagates the raw
predicted probability, a continuous value in [0, 1]. Training and inference
therefore see different context distributions.

This is the leading suspect for CC scoring below BR (-0.017 mAP) while PCC —
which shares the identical fitted chains but enumerates *hard* 0/1 prefixes —
scores above it (+0.008). If the mismatch is the cause, switching CC's inference
to hard context (probabilistic=False) should recover most of the gap.

The ablation is inference-only and perfectly controlled: for each seed the chain
is fitted ONCE and then queried twice, so the two arms differ in nothing but the
context representation. This also halves the compute versus fitting twice.

Writes to results/analysis_v2/cc_context/ — creates new files, overwrites nothing.

Usage:
    python scripts/cc_context_ablation.py --device cuda:1
    python scripts/cc_context_ablation.py --device cuda:1 --chunk_size 1000
"""
import argparse
import datetime
import gc
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS, load_dataset
from src.data.preds_cache import ANALYSIS_LABELS, SEEDS, label_indices, load_test_probs
from src.models import ClassifierChain
from src.utils import RunLogger, set_global_seeds
from src.utils.metrics import compute_paired_bootstrap_delta, fast_ap

DATA_DIR = REPO_ROOT / "data" / "final"
OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "cc_context"
PREDS_OUT = OUT_DIR / "preds"

CHILD_IDX = label_indices(ANALYSIS_LABELS)


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _vram(tag, idx=0):
    try:
        import torch
        if torch.cuda.is_available():
            print(f"    [VRAM] {tag}: {torch.cuda.memory_allocated(idx)/1e9:.2f} GB",
                  flush=True)
    except Exception:
        pass


def load_train_test():
    X1, Y1, feat = load_dataset(str(DATA_DIR / "final_fold1.csv"))
    X2, Y2, _ = load_dataset(str(DATA_DIR / "final_fold2.csv"), feature_cols=feat)
    X_te, Y_te, _ = load_dataset(str(DATA_DIR / "final_test.csv"), feature_cols=feat)
    X_tr = np.concatenate([X1, X2], axis=0)
    Y_tr = np.concatenate([Y1, Y2], axis=0)
    _free(X1, X2, Y1, Y2)
    return X_tr, Y_tr, X_te, Y_te


def predict_chunked(model, X, chunk_size):
    out = []
    for s in range(0, len(X), chunk_size):
        out.append(np.asarray(model.predict_proba(X[s:s + chunk_size]), dtype=np.float32))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--n_estimators", type=int, default=16)
    ap.add_argument("--n_bootstraps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_OUT.mkdir(parents=True, exist_ok=True)
    set_global_seeds(args.seeds[0])

    print("Loading data ...", flush=True)
    X_train, Y_train, X_test, Y_test = load_train_test()
    print(f"  Train {X_train.shape}  Test {X_test.shape}\n", flush=True)

    probs = {"soft": [], "hard": []}
    rows, timings = [], []

    for seed in args.seeds:
        print(f"{'='*60}\n  Seed {seed}\n{'='*60}", flush=True)
        _vram("before fit")

        model = ClassifierChain(
            probabilistic=True,           # affects inference only
            device=args.device,
            n_estimators=args.n_estimators,
            seed=seed,
            ignore_pretraining_limits=True,
        )
        t0 = time.perf_counter()
        model.fit(X_train, Y_train)
        t_fit = time.perf_counter() - t0
        print(f"  fit: {t_fit:.1f}s   chain order: "
              f"{[LABEL_COLS[i] for i in model.order_]}", flush=True)
        _vram("after fit")

        # Same fitted chain, two inference modes.
        seed_probs = {}
        for mode, flag in (("soft", True), ("hard", False)):
            model.probabilistic = flag
            t0 = time.perf_counter()
            p = predict_chunked(model, X_test, args.chunk_size)
            t_inf = time.perf_counter() - t0
            seed_probs[mode] = p
            probs[mode].append(p)
            timings.append({"seed": seed, "mode": mode,
                            "fit_time_s": round(t_fit, 2),
                            "infer_time_s": round(t_inf, 2)})
            print(f"  infer [{mode}]: {t_inf:.1f}s", flush=True)

        _free(model)
        _vram("after free")

        for mode in ("soft", "hard"):
            np.save(PREDS_OUT / f"cc_{mode}_seed{seed}.npy", {
                "strategy": f"CC_{mode}",
                "seed": seed,
                "label_cols": LABEL_COLS,
                "test_probs": seed_probs[mode],
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(seed_probs[mode], dtype=np.float32).tobytes()
                ).hexdigest(),
            })

        for mode in ("soft", "hard"):
            p = seed_probs[mode]
            rec = {"seed": seed, "mode": mode}
            for j, lbl in enumerate(LABEL_COLS):
                rec[f"AP_{lbl}"] = fast_ap(Y_test[:, j], p[:, j])
            rec["mAP5"] = float(np.mean([rec[f"AP_{l}"] for l in LABEL_COLS]))
            rec["mAP4"] = float(np.mean([rec[f"AP_{l}"] for l in ANALYSIS_LABELS]))
            rows.append(rec)
            print(f"  [{mode}] mAP5={rec['mAP5']:.4f}  mAP4={rec['mAP4']:.4f}", flush=True)

        _free(seed_probs)

    df = pd.DataFrame(rows)
    df_mean = df.groupby("mode").mean(numeric_only=True).drop(columns=["seed"])

    # ── reference points from the cached benchmark ────────────────────────────
    cached = load_test_probs(["BR", "CC", "PCC"], args.seeds)
    ref = {}
    for name, plist in cached.items():
        per_label = [
            [fast_ap(Y_test[:, j], p[:, j]) for j in range(len(LABEL_COLS))]
            for p in plist
        ]
        arr = np.array(per_label).mean(axis=0)
        ref[name] = {
            "mAP5": float(arr.mean()),
            "mAP4": float(arr[CHILD_IDX].mean()),
        }

    print("\n" + "=" * 78)
    print("  CC CONTEXT ABLATION — identical fitted chains, inference mode varied")
    print("=" * 78)
    print(df_mean[["mAP5", "mAP4"] +
                  [f"AP_{l}" for l in LABEL_COLS]].round(4).to_string())
    print("\n  Cached reference points (same seeds):")
    for k, v in ref.items():
        print(f"    {k:4s}  mAP5={v['mAP5']:.4f}  mAP4={v['mAP4']:.4f}")

    soft_m, hard_m = df_mean.loc["soft"], df_mean.loc["hard"]
    print(f"\n  hard - soft        : mAP5 {hard_m.mAP5 - soft_m.mAP5:+.4f}   "
          f"mAP4 {hard_m.mAP4 - soft_m.mAP4:+.4f}")
    print(f"  hard - BR (cached) : mAP5 {hard_m.mAP5 - ref['BR']['mAP5']:+.4f}   "
          f"mAP4 {hard_m.mAP4 - ref['BR']['mAP4']:+.4f}")
    print(f"  soft - BR (cached) : mAP5 {soft_m.mAP5 - ref['BR']['mAP5']:+.4f}   "
          f"mAP4 {soft_m.mAP4 - ref['BR']['mAP4']:+.4f}")

    # ── paired bootstrap: hard vs soft, and hard vs cached BR ─────────────────
    print("\n  Paired bootstrap over test peptides ...", flush=True)
    boot = []
    for tag, cols in (("mAP5", list(range(len(LABEL_COLS)))), ("mAP4", CHILD_IDX)):
        r = compute_paired_bootstrap_delta(
            Y_test[:, cols], [p[:, cols] for p in probs["soft"]],
            [p[:, cols] for p in probs["hard"]],
            n_bootstraps=args.n_bootstraps,
        )
        boot.append({"comparison": "CC_hard - CC_soft", "scope": tag, **r})

        r = compute_paired_bootstrap_delta(
            Y_test[:, cols], [p[:, cols] for p in cached["BR"]],
            [p[:, cols] for p in probs["hard"]],
            n_bootstraps=args.n_bootstraps,
        )
        boot.append({"comparison": "CC_hard - BR", "scope": tag, **r})

    df_boot = pd.DataFrame(boot)
    show = df_boot[["comparison", "scope", "delta", "ci_lower",
                    "ci_upper", "significant"]].copy()
    for c in ["delta", "ci_lower", "ci_upper"]:
        show[c] = show[c].map(lambda v: f"{v:+.4f}")
    print(show.to_string(index=False))

    df.to_csv(OUT_DIR / "cc_context_per_seed.csv", index=False)
    df_mean.round(6).to_csv(OUT_DIR / "cc_context_mean.csv")
    df_boot.to_csv(OUT_DIR / "cc_context_paired_bootstrap.csv", index=False)
    pd.DataFrame(timings).to_csv(OUT_DIR / "cc_context_timings.csv", index=False)
    (OUT_DIR / "cc_context_summary.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(),
        "generated_by": "scripts/cc_context_ablation.py",
        "note": ("Chain fitted once per seed; probabilistic flag toggled between "
                 "the two predict_proba calls, so the arms share identical fits."),
        "device": args.device,
        "n_estimators": args.n_estimators,
        "chunk_size": args.chunk_size,
        "seeds": args.seeds,
        "n_bootstraps": args.n_bootstraps,
        "results_mean": df_mean.to_dict(),
        "cached_reference": ref,
    }, indent=2, default=float))

    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    args_preview = sys.argv
    with RunLogger(
        script_name="cc_context_ablation.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=10.0,
        extra_meta={"argv": args_preview},
    ):
        main()
