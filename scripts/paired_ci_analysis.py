"""
Corrected interval estimates and significance tests for the strategy benchmark.

Two problems with the intervals in results/strategies_benchmark/benchmark_summary.csv
are fixed here, both by recomputing from the cached probabilities — no refitting:

1. The reported "mAP 95% CI" was produced by vstacking the 3 seeds' predictions
   into one pool and bootstrapping 3N rows from it, which counts each test
   peptide three times as if independent and shrinks the interval by ~sqrt(3).
   Here one resample of the N real test rows is drawn per iteration and the
   macro-AP is averaged across seeds *inside* that resample.

2. The "Wilcoxon p" column compared strategies across 3 seeds, where the test
   cannot return p < 0.25 for any effect size — every row necessarily read
   0.250. Here strategy differences use a paired bootstrap over the 16,489 test
   peptides, with all strategies scored on identical resamples so the shared
   test-set sampling variance cancels.

Macro averages are reported over both label sets: all five labels (mAP-5, the
convention the existing tables use) and the four real activities (mAP-4), which
excludes the deterministic Antimicrobial OR-parent whose AP of ~0.98 lifts every
strategy uniformly.

Writes to results/analysis_v2/paired_ci/ — creates new files, overwrites nothing.

Usage:
    python scripts/paired_ci_analysis.py
    python scripts/paired_ci_analysis.py --n_bootstraps 5000
"""
import argparse
import datetime
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS
from src.data.preds_cache import (
    ANALYSIS_LABELS,
    SEEDS,
    STRATEGIES,
    load_labels,
    load_oof_probs,
    load_test_probs,
)
from src.utils import RunLogger
from src.utils.metrics import (
    bootstrap_strategy_comparison,
    compute_ap,
    compute_bootstrap_ci,
    compute_honest_f1,
    compute_subset_accuracy,
    find_optimal_thresholds,
)

REFERENCE = "BR"
OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "paired_ci"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def threshold_metrics(Y_train, Y_test, oof_probs, test_probs):
    """
    Honest-F1 and subset accuracy per strategy, averaged over seeds.

    Subset accuracy is reported both at the fixed 0.5 cut (as the original
    tables did) and at the honest per-label thresholds, which matters because a
    0.5 cut predicts the negative class almost everywhere for Antiparasitic.
    """
    rows = {}
    for strat in test_probs:
        hf1, sa_half, sa_honest = [], [], []
        for k, _seed in enumerate(SEEDS):
            p_oof = oof_probs[strat][k]
            p_test = test_probs[strat][k]
            _, macro_hf1, thr = compute_honest_f1(Y_train, p_oof, Y_test, p_test)
            hf1.append(macro_hf1)
            sa_half.append(compute_subset_accuracy(Y_test, p_test, threshold=0.5))
            sa_honest.append(compute_subset_accuracy(Y_test, p_test, thresholds=thr))
        rows[strat] = {
            "honest_f1_mean": float(np.mean(hf1)),
            "honest_f1_std": float(np.std(hf1)),
            "subset_acc_at_0.5_mean": float(np.mean(sa_half)),
            "subset_acc_at_honest_tau_mean": float(np.mean(sa_honest)),
        }
    return rows


def ci_method_comparison(Y_test, test_probs, n_bootstraps, seed):
    """
    Quantify the pooling bug: old pooled-across-seeds CI vs the corrected one.

    Reported so the change in the tables is auditable rather than silent.
    """
    rows = []
    for strat, probs in test_probs.items():
        Y_pool = np.vstack([Y_test] * len(probs))
        P_pool = np.vstack(probs)
        _, old_lo, old_hi = compute_bootstrap_ci(
            Y_pool, P_pool, metric_fn=compute_ap,
            n_bootstraps=n_bootstraps, seed=seed,
        )
        rows.append({
            "Strategy": strat,
            "pooled_ci_lower": old_lo,
            "pooled_ci_upper": old_hi,
            "pooled_ci_width": old_hi - old_lo,
        })
        del Y_pool, P_pool
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the bootstrap resampler (not a model seed)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth and cached predictions ...", flush=True)
    Y_train, Y_test = load_labels()
    test_probs = load_test_probs(STRATEGIES, SEEDS)
    oof_probs = load_oof_probs(STRATEGIES, SEEDS)
    print(f"  Y_train {Y_train.shape}  Y_test {Y_test.shape}  "
          f"{len(STRATEGIES)} strategies x {len(SEEDS)} seeds\n", flush=True)

    macro_subsets = {"mAP5": list(LABEL_COLS), "mAP4": list(ANALYSIS_LABELS)}

    print(f"Shared-resample paired bootstrap ({args.n_bootstraps} resamples, "
          f"reference={REFERENCE}) ...", flush=True)
    res = bootstrap_strategy_comparison(
        y_true=Y_test,
        strategy_probs=test_probs,
        reference=REFERENCE,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        label_names=LABEL_COLS,
        macro_subsets=macro_subsets,
    )
    print("  done.\n", flush=True)

    print("Threshold-dependent metrics (honest tau vs 0.5) ...", flush=True)
    thr_metrics = threshold_metrics(Y_train, Y_test, oof_probs, test_probs)

    print("Old pooled-CI reproduction for the audit trail ...", flush=True)
    df_old = ci_method_comparison(Y_test, test_probs, args.n_bootstraps, args.seed)

    # ── headline table ────────────────────────────────────────────────────────
    rows = []
    for strat in STRATEGIES:
        ps = res["per_strategy"][strat]
        d = res["deltas_vs_reference"].get(strat)
        row = {
            "Strategy": strat,
            "mAP-5": f"{ps['macro']['mAP5']['AP']:.4f}",
            "mAP-5 95% CI": _fmt_ci(ps["macro"]["mAP5"]["ci_lower"],
                                    ps["macro"]["mAP5"]["ci_upper"]),
            "mAP-4": f"{ps['macro']['mAP4']['AP']:.4f}",
            "mAP-4 95% CI": _fmt_ci(ps["macro"]["mAP4"]["ci_lower"],
                                    ps["macro"]["mAP4"]["ci_upper"]),
        }
        if d is None:
            row.update({
                "Δ mAP-5 vs BR": "—", "Δ mAP-5 95% CI": "—", "Δ mAP-5 sig": "—",
                "Δ mAP-4 vs BR": "—", "Δ mAP-4 95% CI": "—", "Δ mAP-4 sig": "—",
            })
        else:
            for key, tag in (("mAP5", "mAP-5"), ("mAP4", "mAP-4")):
                m = d["macro"][key]
                row[f"Δ {tag} vs BR"] = f"{m['delta_AP']:+.4f}"
                row[f"Δ {tag} 95% CI"] = _fmt_ci(m["ci_lower"], m["ci_upper"])
                row[f"Δ {tag} sig"] = "yes" if m["significant"] else "no"
        tm = thr_metrics[strat]
        row["Honest-F1"] = f"{tm['honest_f1_mean']:.4f}"
        row["Subset Acc @0.5"] = f"{tm['subset_acc_at_0.5_mean']:.4f}"
        row["Subset Acc @τ*"] = f"{tm['subset_acc_at_honest_tau_mean']:.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)

    # ── per-label deltas ──────────────────────────────────────────────────────
    lab_rows = []
    for strat, d in res["deltas_vs_reference"].items():
        for lbl, v in d["per_label"].items():
            lab_rows.append({
                "Strategy": strat,
                "Label": lbl,
                "AP": res["per_strategy"][strat]["per_label"][lbl]["AP"],
                "AP_BR": res["per_strategy"][REFERENCE]["per_label"][lbl]["AP"],
                "delta_AP": v["delta_AP"],
                "ci_lower": v["ci_lower"],
                "ci_upper": v["ci_upper"],
                "pseudo_p": v["pseudo_p"],
                "significant": v["significant"],
            })
    df_lab = pd.DataFrame(lab_rows)

    # ── CI width comparison ───────────────────────────────────────────────────
    df_ci = df_old.copy()
    df_ci["correct_ci_lower"] = [
        res["per_strategy"][s]["macro"]["mAP5"]["ci_lower"] for s in df_ci.Strategy
    ]
    df_ci["correct_ci_upper"] = [
        res["per_strategy"][s]["macro"]["mAP5"]["ci_upper"] for s in df_ci.Strategy
    ]
    df_ci["correct_ci_width"] = df_ci.correct_ci_upper - df_ci.correct_ci_lower
    df_ci["width_ratio_correct_over_pooled"] = (
        df_ci.correct_ci_width / df_ci.pooled_ci_width
    )

    # ── save ──────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().isoformat()
    payload = {
        "generated_at": ts,
        "generated_by": "scripts/paired_ci_analysis.py",
        "preds_cache": "data/preds_prob/ (see MANIFEST.json)",
        "seeds": SEEDS,
        "reference": REFERENCE,
        "n_bootstraps": args.n_bootstraps,
        "bootstrap_seed": args.seed,
        "threshold_metrics": thr_metrics,
        "bootstrap": res,
    }
    (OUT_DIR / "paired_bootstrap.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    df.to_csv(OUT_DIR / "summary_paired_ci.csv", index=False)
    df_lab.to_csv(OUT_DIR / "per_label_deltas.csv", index=False)
    df_ci.to_csv(OUT_DIR / "ci_method_comparison.csv", index=False)

    # ── report ────────────────────────────────────────────────────────────────
    pd.set_option("display.width", 250)
    print("\n" + "=" * 110)
    print(f"  CORRECTED BENCHMARK STATISTICS — {args.n_bootstraps} shared resamples "
          f"of n={res['n_test']} test peptides")
    print("=" * 110)
    print(df.to_string(index=False))

    print("\n\n--- Per-label Δ AP vs BR (paired bootstrap) ---")
    show = df_lab[["Strategy", "Label", "delta_AP", "ci_lower", "ci_upper",
                   "pseudo_p", "significant"]].copy()
    for c in ["delta_AP", "ci_lower", "ci_upper"]:
        show[c] = show[c].map(lambda v: f"{v:+.4f}")
    show["pseudo_p"] = show["pseudo_p"].map(lambda v: f"{v:.4f}")
    print(show.to_string(index=False))

    print("\n\n--- CI method comparison (mAP-5) ---")
    print(df_ci.round(4).to_string(index=False))
    print(f"\n  Mean width ratio (correct / pooled): "
          f"{df_ci.width_ratio_correct_over_pooled.mean():.3f}   "
          f"(sqrt(3) = {np.sqrt(3):.3f} is the expected inflation from pooling 3 seeds)")

    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    with RunLogger(
        script_name="paired_ci_analysis.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=30.0,
        extra_meta={"argv": sys.argv},
    ):
        main()
