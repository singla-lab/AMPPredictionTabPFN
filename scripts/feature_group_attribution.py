"""
T8: which feature families carry the signal, per label.

Three complementary views, because each answers a different question and they
disagree in informative ways when features are redundant:

  permutation   Shuffle a group's columns in the TEST matrix and re-infer with
                the SAME fitted model. Measures how much the fitted model relies
                on that group. Redundant groups look unimportant here: if two
                groups carry the same information, destroying either alone costs
                little.
  leave_one_out Re-fit without the group. Measures what is lost when the group
                is unavailable from the start. Also insensitive to redundancy,
                for the same reason.
  group_only    Re-fit on that group alone. Measures the group's standalone
                sufficiency, which is the view that survives redundancy.

Reported as AP drop (permutation, leave_one_out) or absolute AP (group_only),
per label and macro, against a baseline fitted on all 330 features.

NOT SHAP. KernelSHAP over 330 features would need thousands of TabPFN forward
passes per peptide; at ~9 min per label per full-test inference that is not
affordable. Group permutation importance answers the same "which inputs matter"
question at a tractable cost, and is labelled as what it is.

Cost control: inference dominates (fit is ~40 s, inference ~9 min per label on
the full 16,489-row test set), so all arms run on a stratified subsample of the
test set. The subsample is drawn ONCE and reused by every arm, so all arms are
mutually comparable; only comparisons against full-test numbers elsewhere in the
results need care.

Writes to results/analysis_v2/feature_attribution/. Reads data/final/ only;
nothing under data/preds_prob/ is touched.

Usage:
    python scripts/feature_group_attribution.py --device cuda:1
"""
import argparse
import datetime
import gc
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS, load_dataset
from src.data.preds_cache import ANALYSIS_LABELS
from src.models import BinaryRelevance
from src.utils import RunLogger, set_global_seeds
from src.utils.metrics import fast_ap

OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "feature_attribution"
DATA = REPO_ROOT / "data" / "final"

CTD_PROPS = ["Hydrophobicity", "Volume", "Polarity", "Polarizability",
             "Charge", "SecondaryStruct", "SolventAccessibility"]
PHYSCHEM = ["Length", "Charge", "ChargeDensity", "pI", "InstabilityInd",
            "Aromaticity", "AliphaticInd", "BomanInd", "HydrophRatio",
            "HydrophobicMoment"]


def build_groups(feat):
    """
    Map each feature to a top-level family, plus the 7 CTD sub-properties.

    'Charge' is ambiguous: it is both a CTD property prefix (Charge_C_1, ...)
    and a bare physicochemical scalar. The CTD test requires the underscore, so
    the bare column falls through to PhysChem, which is correct.
    """
    idx = {c: i for i, c in enumerate(feat)}
    top = {k: [] for k in ["ESM", "CTD", "AAC", "AAIndex", "PhysChem",
                           "Motif", "Density", "Cleavage"]}
    sub = {f"CTD:{p}": [] for p in CTD_PROPS}
    for c in feat:
        i = idx[c]
        ctd = next((p for p in CTD_PROPS if c.startswith(p + "_")), None)
        if c.startswith("ESM"):
            top["ESM"].append(i)
        elif ctd:
            top["CTD"].append(i)
            sub[f"CTD:{ctd}"].append(i)
        elif c.startswith("AAC_"):
            top["AAC"].append(i)
        elif c.startswith("AAIndex"):
            top["AAIndex"].append(i)
        elif c.startswith("motif"):
            top["Motif"].append(i)
        elif c.startswith("dens"):
            top["Density"].append(i)
        elif c.startswith("cleavage"):
            top["Cleavage"].append(i)
        elif c in PHYSCHEM:
            top["PhysChem"].append(i)
        else:
            raise ValueError(f"unassigned feature: {c}")
    covered = sorted(i for v in top.values() for i in v)
    assert covered == list(range(len(feat))), "feature partition is not exhaustive"
    return top, sub


def stratified_subsample(Y, n, seed):
    """
    Keep every positive of the rarest labels, fill the rest at random.

    Antiparasitic has so few positives that a uniform draw would leave single
    digits; taking all of them keeps the column from being degenerate, at the
    cost of a subsample whose prevalence is not the test set's. Prevalences are
    recorded in the output so this is visible.
    """
    rng = np.random.RandomState(seed)
    keep = set()
    order = np.argsort(Y.sum(axis=0))          # rarest label first
    for j in order:
        pos = np.flatnonzero(Y[:, j] == 1)
        if len(pos) <= 200:
            keep.update(pos.tolist())
    rest = np.setdiff1d(np.arange(len(Y)), np.fromiter(keep, int))
    need = max(0, n - len(keep))
    keep.update(rng.choice(rest, size=min(need, len(rest)), replace=False).tolist())
    return np.sort(np.fromiter(keep, int))


def aps(model, X, Y):
    p = model.predict_proba(X)
    return [float(fast_ap(Y[:, j], p[:, j])) for j in range(Y.shape[1])]


def summarise(row_aps):
    return {**{f"AP_{l}": a for l, a in zip(LABEL_COLS, row_aps)},
            "mAP5": float(np.mean(row_aps)),
            "mAP4": float(np.mean([row_aps[LABEL_COLS.index(l)]
                                   for l in ANALYSIS_LABELS]))}


def free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_test", type=int, default=4000)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--arms", nargs="+",
                    default=["permutation", "leave_one_out", "group_only"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seeds(args.seed)

    X1, Y1, feat = load_dataset(str(DATA / "final_fold1.csv"))
    X2, Y2, _ = load_dataset(str(DATA / "final_fold2.csv"), feature_cols=feat)
    Xte, Yte, _ = load_dataset(str(DATA / "final_test.csv"), feature_cols=feat)
    Xtr = np.concatenate([X1, X2]); Ytr = np.concatenate([Y1, Y2])
    del X1, X2, Y1, Y2
    top, sub = build_groups(feat)
    groups = {**top, **sub}
    print(f"features {len(feat)}  groups " +
          ", ".join(f"{k}={len(v)}" for k, v in top.items()), flush=True)

    sel = stratified_subsample(Yte, args.n_test, args.seed)
    Xs, Ys = Xte[sel], Yte[sel]
    print(f"test subsample {len(sel):,} / {len(Yte):,}; positives " +
          ", ".join(f"{l}={int(Ys[:, j].sum())}" for j, l in enumerate(LABEL_COLS)),
          flush=True)

    ckpt = OUT_DIR / "_checkpoint.json"
    done = json.loads(ckpt.read_text()) if ckpt.exists() else {}

    def record(key, payload):
        done[key] = payload
        ckpt.write_text(json.dumps(done, indent=2))

    def fit(cols=None):
        # BinaryRelevance takes `seed`, not `random_state`, and does its own
        # batching inside predict_proba -- there is no chunk_size argument.
        m = BinaryRelevance(device=args.device, seed=args.seed,
                            n_estimators=16, ignore_pretraining_limits=True)
        m.fit(Xtr if cols is None else Xtr[:, cols], Ytr)
        return m

    # ---- baseline -----------------------------------------------------------
    if "baseline" not in done:
        t0 = datetime.datetime.now()
        base_model = fit()
        base = aps(base_model, Xs, Ys)
        record("baseline", summarise(base))
        print(f"baseline mAP5={done['baseline']['mAP5']:.4f} "
              f"({(datetime.datetime.now()-t0).total_seconds()/60:.1f} min)", flush=True)
    else:
        base_model = None
    base = [done["baseline"][f"AP_{l}"] for l in LABEL_COLS]

    # ---- permutation --------------------------------------------------------
    if "permutation" in args.arms:
        if base_model is None:
            base_model = fit()
        for g, cols in groups.items():
            key = f"perm::{g}"
            if key in done:
                continue
            reps = []
            for r in range(args.n_repeats):
                rng = np.random.RandomState(1000 * args.seed + r)
                Xp = Xs.copy()
                Xp[:, cols] = Xp[rng.permutation(len(Xp))[:, None], np.array(cols)]
                reps.append(aps(base_model, Xp, Ys))
                free()
            arr = np.array(reps)
            record(key, {"n_features": len(cols), "n_repeats": args.n_repeats,
                         "mean": summarise(arr.mean(axis=0)),
                         "std_mAP5": float(arr.mean(axis=1).std(ddof=1))
                         if args.n_repeats > 1 else 0.0})
            print(f"  perm {g:24s} mAP5 {done[key]['mean']['mAP5']:.4f} "
                  f"(drop {base_and(base)-done[key]['mean']['mAP5']:+.4f})", flush=True)
        del base_model
        free()

    # ---- leave-one-group-out and group-only ---------------------------------
    for arm, tag in [("leave_one_out", "logo"), ("group_only", "only")]:
        if arm not in args.arms:
            continue
        for g, cols in top.items():          # top-level families only
            key = f"{tag}::{g}"
            if key in done:
                continue
            use = ([i for i in range(len(feat)) if i not in set(cols)]
                   if arm == "leave_one_out" else cols)
            if not use:
                continue
            m = fit(use)
            r = aps(m, Xs[:, use], Ys)
            record(key, {"n_features": len(use), **summarise(r)})
            print(f"  {tag} {g:24s} mAP5 {done[key]['mAP5']:.4f} "
                  f"({len(use)} feats)", flush=True)
            del m
            free()

    # ---- tables -------------------------------------------------------------
    rows = []
    for k, v in done.items():
        if k == "baseline":
            continue
        arm, g = k.split("::")
        m = v["mean"] if arm == "perm" else v
        rows.append({"arm": arm, "group": g, "n_features": v["n_features"],
                     **{c: m[c] for c in m if c.startswith("AP_")
                        or c in ("mAP5", "mAP4")}})
    df = pd.DataFrame(rows)
    for c in ["mAP5", "mAP4"] + [f"AP_{l}" for l in LABEL_COLS]:
        df[f"drop_{c}"] = done["baseline"][c] - df[c]
    df = df.sort_values(["arm", "drop_mAP5"], ascending=[True, False])
    df.to_csv(OUT_DIR / "feature_group_attribution.csv", index=False)

    (OUT_DIR / "feature_attribution_summary.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(),
        "generated_by": "scripts/feature_group_attribution.py",
        "method": "group permutation importance + LOGO/group-only refits (NOT SHAP)",
        "model": "BinaryRelevance (TabPFN)", "seed": args.seed,
        "n_test_subsample": int(len(sel)), "n_test_full": int(len(Yte)),
        "subsample_positives": {l: int(Ys[:, j].sum())
                                for j, l in enumerate(LABEL_COLS)},
        "full_test_positives": {l: int(Yte[:, j].sum())
                                for j, l in enumerate(LABEL_COLS)},
        "n_repeats_permutation": args.n_repeats,
        "baseline": done["baseline"],
        "group_sizes": {k: len(v) for k, v in groups.items()},
    }, indent=2, default=float))
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved -> {OUT_DIR}")


def base_and(b):
    return float(np.mean(b))


if __name__ == "__main__":
    with RunLogger(script_name="feature_group_attribution.py",
                   log_dir=REPO_ROOT / "logs", sample_interval_s=30.0,
                   extra_meta={"argv": sys.argv}):
        main()
