"""
Experiment 19: fold-matched training protocol vs ESCAPE.

The confound. Our strategies train on fold1 + fold2 (65,870 rows). Each ESCAPE
checkpoint trains on ONE fold (~32,900 rows) and the released model averages two
such checkpoints. So the reported margin mixes two effects: the model, and twice
the training data. Both are the same size in this dataset:

    halving the training set costs   0.0501 mAP-5   (Section 9.1, BR 31,250 vs 65,870)
    our paired margin over ESCAPE    0.0543 mAP-5   (Section 20.7, BR)

The existing n = 31,250 scaling cell does not settle it: it draws a random subset
spanning both folds rather than a fold partition, and it is a single model, whereas
ESCAPE's number is a two-checkpoint ensemble worth +0.0522 to them. The matched
protocol is therefore two half-data models, averaged.

Design, per strategy in {BR, LP} and seed in {42, 1665, 8914}:
    fold1-only   fit on final_fold1.csv (32,948 rows), predict test
    fold2-only   fit on final_fold2.csv (32,922 rows), predict test
    ensemble     (P_fold1 + P_fold2) / 2, matching ESCAPE's averaging of sigmoids

Hyperparameters identical to Section 5. All reported metrics are threshold-free, so
no OOF pass is run and no threshold-dependent metric is reported.

DECISION RULE, fixed before the numbers exist (from the task spec):
  * fold-matched ensemble still beats ESCAPE, significant -> headline survives in its
    strongest form; report both protocols, lead with the matched one; then run PCC.
  * indistinguishable from ESCAPE -> restate as "matches ESCAPE under its own training
    protocol, exceeds it with the full training set"; do not run PCC.
  * worse than ESCAPE -> SOTA claim withdrawn; flag immediately.

Writes to results/analysis_v2/fold_matched/. data/preds_prob/ is not touched.

Usage:
    python scripts/fold_matched_protocol.py --device cuda:0
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

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import LABEL_COLS, load_dataset, read_table
from src.data.preds_cache import ANALYSIS_LABELS, load_test_probs
from src.models import (BinaryRelevance, ClassifierChain, EnsembleClassifierChain,
                        LabelPowerset, ProbabilisticClassifierChain)
from src.utils import RunLogger, set_global_seeds
from src.utils.metrics import compute_paired_bootstrap_delta, fast_ap

DATA = REPO / "data" / "final"
OUT_DIR = REPO / "results" / "analysis_v2" / "fold_matched"
PRED_DIR = OUT_DIR / "preds"
ESC = REPO / "results" / "analysis_v2" / "escape_repro"
SEEDS = [42, 1665, 8914]
STRATEGIES = {"BR": BinaryRelevance, "LP": LabelPowerset,
              "CC": ClassifierChain, "ECC": EnsembleClassifierChain,
              "PCC": ProbabilisticClassifierChain}
CH = [LABEL_COLS.index(c) for c in ANALYSIS_LABELS]

# Full-data references from Section 5 (3-seed means), for the data-scale cost.
FULL_DATA = {"BR": (0.7701, 0.7174), "LP": (0.7779, 0.7271),
             "CC": (0.7527, 0.6970), "ECC": (0.7689, 0.7166),
             "PCC": (0.7776, 0.7268)}


def sha(p):
    return hashlib.sha256(np.ascontiguousarray(p, dtype=np.float32).tobytes()).hexdigest()


def save(strategy, seed, arm, probs):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{strategy.lower()}_{arm}_seed{seed}.npy"
    np.save(path, {"strategy": strategy, "seed": seed, "arm": arm,
                   "label_cols": list(LABEL_COLS), "test_probs": probs,
                   "sha256": sha(probs)}, allow_pickle=True)
    return path


def load_saved(strategy, seed, arm):
    path = PRED_DIR / f"{strategy.lower()}_{arm}_seed{seed}.npy"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True).item()
    assert list(d["label_cols"]) == list(LABEL_COLS), f"label order mismatch in {path}"
    P = np.asarray(d["test_probs"], dtype=float)
    assert sha(P) == d["sha256"], f"SHA-256 mismatch in {path}"
    return P


def aps(Y, P):
    a = [float(fast_ap(Y[:, j], P[:, j])) for j in range(Y.shape[1])]
    return a, float(np.mean(a)), float(np.mean([a[j] for j in CH]))


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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--strategies", nargs="+", default=["LP", "BR"])
    ap.add_argument("--n_bootstraps", type=int, default=2000)
    # Fit and cache predictions only, skipping every aggregate table. A partial
    # run (one strategy, or a subset of seeds) would otherwise overwrite the
    # shared CSVs with only its own rows and destroy the other strategies'
    # results. Use this to parallelise fits, then run once with every strategy
    # to build the tables from the cache.
    ap.add_argument("--fits_only", action="store_true")
    # Restrict which fold arms this process fits, so 6 fits can be split evenly
    # across 2 GPUs (3 each) rather than by seed (4/2). Only meaningful with
    # --fits_only: the ensemble arm needs both folds and is built by the final
    # full pass from the cache.
    ap.add_argument("--folds", nargs="+", default=["fold1", "fold2"],
                    choices=["fold1", "fold2"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X1, Y1, feat = load_dataset(str(DATA / "final_fold1.csv"))
    X2, Y2, _ = load_dataset(str(DATA / "final_fold2.csv"), feature_cols=feat)
    Xte, Yte, _ = load_dataset(str(DATA / "final_test.csv"), feature_cols=feat)
    folds = {"fold1": (X1, Y1), "fold2": (X2, Y2)}
    print(f"fold1 {X1.shape[0]:,}  fold2 {X2.shape[0]:,}  test {Xte.shape[0]:,}", flush=True)

    # ── fit the six half-data models ─────────────────────────────────────────
    probs = {}
    rows = []
    for name in args.strategies:
        cls = STRATEGIES[name]
        for seed in args.seeds:
            for arm in args.folds:
                cached = load_saved(name, seed, arm)
                if cached is not None:
                    probs[(name, seed, arm)] = cached
                    print(f"  {name} seed{seed} {arm}: cached", flush=True)
                    continue
                set_global_seeds(seed)
                Xf, Yf = folds[arm]
                m = cls(device=args.device, seed=seed, n_estimators=16,
                        ignore_pretraining_limits=True)
                t0 = time.time(); m.fit(Xf, Yf); t_fit = time.time() - t0
                t0 = time.time(); P = m.predict_proba(Xte); t_inf = time.time() - t0
                probs[(name, seed, arm)] = P
                save(name, seed, arm, P)
                _, m5, m4 = aps(Yte, P)
                print(f"  {name} seed{seed} {arm}: fit={t_fit:.0f}s infer={t_inf:.0f}s "
                      f"mAP5={m5:.4f} mAP4={m4:.4f}", flush=True)
                del m; free()
            # The matched arm needs both halves. When this process was asked
            # for only one fold, leave it to the final full pass.
            if not all((name, seed, a) in probs for a in ("fold1", "fold2")):
                print(f"  {name} seed{seed}: only {args.folds} fitted here; "
                      f"ensemble deferred", flush=True)
                continue
            # the matched arm: arithmetic mean of the two half-data models
            pe = (probs[(name, seed, "fold1")] + probs[(name, seed, "fold2")]) / 2.0
            probs[(name, seed, "ensemble")] = pe
            save(name, seed, "ensemble", pe)
            _, m5, m4 = aps(Yte, pe)
            print(f"  {name} seed{seed} ENSEMBLE: mAP5={m5:.4f} mAP4={m4:.4f}", flush=True)

    if args.fits_only:
        print(f"\n--fits_only: {len(probs)} arrays cached under {PRED_DIR}; "
              f"no tables written.", flush=True)
        return

    # ── 3.1 core table, full test set ────────────────────────────────────────
    for name in args.strategies:
        for arm in ["fold1", "fold2", "ensemble"]:
            for seed in args.seeds:
                a, m5, m4 = aps(Yte, probs[(name, seed, arm)])
                rows.append({"strategy": name, "arm": arm, "seed": seed,
                             **{f"AP_{l}": v for l, v in zip(LABEL_COLS, a)},
                             "mAP5": m5, "mAP4": m4})
    per = pd.DataFrame(rows)
    per.to_csv(OUT_DIR / "fold_matched_per_seed.csv", index=False)

    agg = per.groupby(["strategy", "arm"]).agg(
        **{f"AP_{l}": (f"AP_{l}", "mean") for l in LABEL_COLS},
        mAP5_mean=("mAP5", "mean"), mAP5_sd=("mAP5", lambda s: s.std(ddof=1)),
        mAP4_mean=("mAP4", "mean"), mAP4_sd=("mAP4", lambda s: s.std(ddof=1)),
    ).reset_index()
    agg.to_csv(OUT_DIR / "fold_matched_summary.csv", index=False)
    print("\n" + "=" * 78)
    print("  3.1 CORE TABLE - full test set (n = %d), 3-seed mean" % len(Yte))
    print("=" * 78)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    gains = []
    for name in args.strategies:
        g = agg[agg.strategy == name].set_index("arm")
        single = (g.loc["fold1", "mAP5_mean"] + g.loc["fold2", "mAP5_mean"]) / 2
        single4 = (g.loc["fold1", "mAP4_mean"] + g.loc["fold2", "mAP4_mean"]) / 2
        gains.append({
            "strategy": name,
            "mean_single_fold_mAP5": single,
            "fold_ensemble_mAP5": g.loc["ensemble", "mAP5_mean"],
            "ensembling_gain_mAP5": g.loc["ensemble", "mAP5_mean"] - single,
            "full_data_mAP5": FULL_DATA[name][0],
            "data_scale_cost_mAP5": FULL_DATA[name][0] - g.loc["ensemble", "mAP5_mean"],
            "mean_single_fold_mAP4": single4,
            "fold_ensemble_mAP4": g.loc["ensemble", "mAP4_mean"],
            "ensembling_gain_mAP4": g.loc["ensemble", "mAP4_mean"] - single4,
            "full_data_mAP4": FULL_DATA[name][1],
            "data_scale_cost_mAP4": FULL_DATA[name][1] - g.loc["ensemble", "mAP4_mean"],
        })
    gd = pd.DataFrame(gains)
    gd.to_csv(OUT_DIR / "fold_matched_gains.csv", index=False)
    print("\n  ensembling gain and data-scale cost:")
    print(gd.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # ── 3.2 paired bootstrap: fold-ensemble vs full-data ─────────────────────
    cache = load_test_probs(strategies=args.strategies)
    boot = []
    for name in args.strategies:
        ens = [probs[(name, s, "ensemble")] for s in args.seeds]
        full = cache[name]
        for tag, cols in (("mAP5", list(range(5))), ("mAP4", CH)):
            r = compute_paired_bootstrap_delta(
                Yte[:, cols], [p[:, cols] for p in ens], [p[:, cols] for p in full],
                n_bootstraps=args.n_bootstraps)
            boot.append({"comparison": f"{name} full-data - fold-ensemble",
                         "scope": tag, **r})
    bd = pd.DataFrame(boot)
    bd.to_csv(OUT_DIR / "fold_matched_vs_fulldata_bootstrap.csv", index=False)
    print("\n" + "=" * 78)
    print("  3.2 PAIRED BOOTSTRAP - full-data minus fold-ensemble (the confound)")
    print("=" * 78)
    sh = bd[["comparison", "scope", "delta", "ci_lower", "ci_upper", "significant"]].copy()
    for c in ["delta", "ci_lower", "ci_upper"]:
        sh[c] = sh[c].map(lambda v: f"{v:+.4f}")
    print(sh.to_string(index=False))

    # ── 3.3 paired bootstrap vs ESCAPE, on the covered subset ────────────────
    z = np.load(ESC / "escape_test_subset_preds.npz", allow_pickle=True)
    e_hash = z["hashes"].astype(str)
    e_labels = [str(x) for x in z["labels"]]
    e_prob, e_true = z["ensemble"], z["y_true"]
    test_meta = read_table(DATA / "final_test.csv")
    test_meta["Hash"] = test_meta["Hash"].astype(str)
    pos = {h: i for i, h in enumerate(test_meta["Hash"].values)}
    sel = np.array([pos[h] for h in e_hash])
    assert (test_meta["Hash"].values[sel] == e_hash).all(), "row alignment failed"
    col_of = {c: LABEL_COLS.index(c) for c in e_labels}
    Ys = test_meta.loc[sel, e_labels].values.astype(int)
    assert np.array_equal(Ys, e_true.astype(int)), "ground-truth mismatch vs ESCAPE loader"
    print(f"\nESCAPE-covered subset: {len(sel):,} peptides", flush=True)

    def paired_macro(Y, pa, pb, cols, n_boot, seed=42):
        rng = np.random.RandomState(seed)
        obs = (np.mean([fast_ap(Y[:, j], pb[:, j]) for j in cols])
               - np.mean([fast_ap(Y[:, j], pa[:, j]) for j in cols]))
        d = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(Y), len(Y))
            a = np.mean([fast_ap(Y[idx, j], pa[idx, j]) for j in cols])
            b = np.mean([fast_ap(Y[idx, j], pb[idx, j]) for j in cols])
            if np.isfinite(a) and np.isfinite(b):
                d.append(b - a)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta": float(obs), "ci_lower": float(lo), "ci_upper": float(hi),
                "significant": bool(lo > 0 or hi < 0)}

    def paired_label(y, pa, pb, n_boot, seed=42):
        rng = np.random.RandomState(seed)
        obs = fast_ap(y, pb) - fast_ap(y, pa)
        d = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(y), len(y))
            if y[idx].sum() in (0, len(y)):
                continue
            d.append(fast_ap(y[idx], pb[idx]) - fast_ap(y[idx], pa[idx]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta": float(obs), "ci_lower": float(lo), "ci_upper": float(hi),
                "significant": bool(lo > 0 or hi < 0)}

    esc_rows, side = [], []
    for name in args.strategies:
        # seed-average the fold-matched ensemble, then restrict to the subset
        pm = np.mean(np.stack([probs[(name, s, "ensemble")] for s in args.seeds]), axis=0)
        ours = np.column_stack([pm[sel, col_of[c]] for c in e_labels])
        pf = np.mean(np.stack(cache[name]), axis=0)
        ours_full = np.column_stack([pf[sel, col_of[c]] for c in e_labels])
        for j, lab in enumerate(e_labels):
            r = paired_label(Ys[:, j], e_prob[:, j], ours[:, j], args.n_bootstraps)
            esc_rows.append({"strategy": f"{name} fold-matched", "label": lab,
                             "n_pos": int(Ys[:, j].sum()), **r})
        for tag, subset in (("mAP-5", e_labels), ("mAP-4", ANALYSIS_LABELS)):
            cols = [e_labels.index(c) for c in subset]
            r = paired_macro(Ys, e_prob, ours, cols, args.n_bootstraps)
            esc_rows.append({"strategy": f"{name} fold-matched", "label": tag,
                             "n_pos": -1, **r})
        side.append({"model": f"{name} fold-matched",
                     "protocol": "2 models, 1 fold each, averaged",
                     "n_train_per_model": 32935,
                     "mAP5_subset": float(np.mean([fast_ap(Ys[:, j], ours[:, j])
                                                   for j in range(len(e_labels))]))})
        side.append({"model": f"{name} full-data", "protocol": "1 model, both folds",
                     "n_train_per_model": 65870,
                     "mAP5_subset": float(np.mean([fast_ap(Ys[:, j], ours_full[:, j])
                                                   for j in range(len(e_labels))]))})
    ed = pd.DataFrame(esc_rows)
    ed.to_csv(OUT_DIR / "fold_matched_vs_escape.csv", index=False)
    print("\n" + "=" * 78)
    print("  3.3 PAIRED BOOTSTRAP - fold-matched ensemble minus ESCAPE")
    print("=" * 78)
    sh = ed[["strategy", "label", "n_pos", "delta", "ci_lower", "ci_upper",
             "significant"]].copy()
    for c in ["delta", "ci_lower", "ci_upper"]:
        sh[c] = sh[c].map(lambda v: f"{v:+.4f}")
    print(sh.to_string(index=False))

    # ── 3.4 side-by-side ─────────────────────────────────────────────────────
    esc_m5 = float(np.mean([fast_ap(Ys[:, j], e_prob[:, j]) for j in range(len(e_labels))]))
    side.insert(0, {"model": "ESCAPE (reproduced)",
                    "protocol": "2 checkpoints, 1 fold each, averaged",
                    "n_train_per_model": 32935, "mAP5_subset": esc_m5})
    sd = pd.DataFrame(side)
    sd.to_csv(OUT_DIR / "fold_matched_side_by_side.csv", index=False)
    print("\n  3.4 SIDE-BY-SIDE on the %d-peptide subset:" % len(sel))
    print(sd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ── 4. decision rule ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  4. DECISION RULE (fixed before the numbers existed)")
    print("=" * 78)
    verdicts = {}
    for name in args.strategies:
        r = ed[(ed.strategy == f"{name} fold-matched") & (ed.label == "mAP-5")].iloc[0]
        if r.significant and r.delta > 0:
            v = "BEATS ESCAPE (significant) -> headline survives; lead with matched protocol"
        elif not r.significant:
            v = ("INDISTINGUISHABLE from ESCAPE -> restate as 'matches under ESCAPE's own "
                 "protocol, exceeds with full training set'; do NOT run PCC")
        else:
            v = "WORSE than ESCAPE (significant) -> SOTA claim withdrawn"
        verdicts[name] = {"delta_mAP5": float(r.delta),
                          "ci": [float(r.ci_lower), float(r.ci_upper)],
                          "significant": bool(r.significant), "verdict": v}
        print(f"  {name}: delta={r.delta:+.4f} CI=[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}] "
              f"-> {v}")

    json.dump({
        "generated_at": datetime.datetime.now().isoformat(),
        "generated_by": "scripts/fold_matched_protocol.py",
        "n_test": int(len(Yte)), "n_escape_subset": int(len(sel)),
        "seeds": args.seeds, "strategies": args.strategies,
        "n_train_fold1": int(X1.shape[0]), "n_train_fold2": int(X2.shape[0]),
        "escape_mAP5_subset": esc_m5,
        "decision_rule_outcome": verdicts,
    }, open(OUT_DIR / "fold_matched_summary.json", "w"), indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    with RunLogger(script_name="fold_matched_protocol.py", log_dir=REPO / "logs",
                   sample_interval_s=30.0, extra_meta={"argv": sys.argv}):
        main()
