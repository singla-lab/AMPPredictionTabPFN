"""
Exploiting the deterministic Antimicrobial OR-parent.

Antimicrobial equals OR(Antibacterial, Antifungal, Antiviral, Antiparasitic) in
every row of every split, yet all five strategies predict it as an ordinary
fifth label. This script quantifies what that costs and what enforcing the
constraint buys, entirely from cached probabilities (no refitting):

  1. Ground-truth verification that the OR relation holds exactly.
  2. Consistency audit — how often each strategy's own output contradicts the
     constraint, in probability space (p_parent >= max p_child) and after
     thresholding (orphan child / empty parent).
  3. Derived parent — replace the predicted Antimicrobial column with max(children),
     noisy-OR, or their mean, and compare AP and honest-F1 against direct prediction.
  4. Reverse projection — constrain the children by the parent (clamp / multiply)
     and measure the effect on the four real activities.

Headline deltas carry paired-bootstrap CIs over the test peptides.

Writes to results/analysis_v2/orgate/ — creates new files, overwrites nothing.

Usage:
    python scripts/orgate_analysis.py
    python scripts/orgate_analysis.py --n_bootstraps 5000
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
    OR_PARENT,
    SEEDS,
    STRATEGIES,
    label_indices,
    load_labels,
    load_oof_probs,
    load_test_probs,
)
from src.utils import RunLogger
from src.utils.metrics import (
    compute_paired_bootstrap_delta,
    fast_ap,
    find_optimal_thresholds,
)
from src.utils.orgate import (
    hard_or,
    hard_violations,
    probability_violations,
    project_children,
    project_parent,
    verify_or_gate,
)
from sklearn.metrics import f1_score

OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "orgate"

CHILD_IDX = label_indices(ANALYSIS_LABELS)
PARENT_IDX = LABEL_COLS.index(OR_PARENT)

PARENT_METHODS = ["max", "noisy_or", "mean_max_noisyor"]
CHILD_METHODS = ["clamp", "multiply"]


def _macro_ap(Y, P, cols):
    return float(np.mean([fast_ap(Y[:, j], P[:, j]) for j in cols]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth and cached predictions ...", flush=True)
    Y_train, Y_test = load_labels()
    test_probs = load_test_probs(STRATEGIES, SEEDS)
    oof_probs = load_oof_probs(STRATEGIES, SEEDS)

    # ── 1. verify the OR relation ─────────────────────────────────────────────
    gt = {
        "train": verify_or_gate(Y_train, CHILD_IDX, PARENT_IDX),
        "test": verify_or_gate(Y_test, CHILD_IDX, PARENT_IDX),
    }
    print("\n--- 1. Ground-truth OR-gate verification ---")
    for split, r in gt.items():
        print(f"  {split:5s} n={r['n_rows']:6d}  child_without_parent={r['child_without_parent']}  "
              f"parent_without_child={r['parent_without_child']}  holds_exactly={r['holds_exactly']}")
    if not all(r["holds_exactly"] for r in gt.values()):
        print("  [WARNING] OR relation does NOT hold exactly; downstream results are not licensed.")

    # ── 2. consistency audit ──────────────────────────────────────────────────
    audit_rows = []
    thresholds_by_strat = {}
    for strat in STRATEGIES:
        for k, seed in enumerate(SEEDS):
            P = test_probs[strat][k]
            thr = find_optimal_thresholds(Y_train, oof_probs[strat][k])
            thresholds_by_strat[(strat, seed)] = thr
            row = {"Strategy": strat, "seed": seed}
            row.update(probability_violations(P, CHILD_IDX, PARENT_IDX))
            row.update(hard_violations(P, thr, CHILD_IDX, PARENT_IDX))
            audit_rows.append(row)
    df_audit = pd.DataFrame(audit_rows)
    df_audit_mean = df_audit.groupby("Strategy").mean(numeric_only=True).drop(columns=["seed"])

    print("\n--- 2. Constraint-violation audit (mean over 3 seeds) ---")
    print(df_audit_mean[[
        "prob_violation_rate", "mean_violation_margin", "max_violation_margin",
        "orphan_child_rate", "empty_parent_rate", "any_violation_rate",
    ]].round(4).to_string())

    # ── 3. derived parent ─────────────────────────────────────────────────────
    parent_rows = []
    for strat in STRATEGIES:
        for k, seed in enumerate(SEEDS):
            P = test_probs[strat][k]
            P_oof = oof_probs[strat][k]
            thr = thresholds_by_strat[(strat, seed)]
            y_true_parent = Y_test[:, PARENT_IDX]

            def _record(variant, p_parent_test, p_parent_oof):
                # Honest threshold re-selected on OOF for each derivation, so the
                # comparison is like-for-like rather than reusing the direct cut.
                oof_col = np.array(P_oof, copy=True)
                oof_col[:, PARENT_IDX] = p_parent_oof
                thr_v = find_optimal_thresholds(Y_train, oof_col)[PARENT_IDX]
                pred = (p_parent_test >= thr_v).astype(int)
                parent_rows.append({
                    "Strategy": strat, "seed": seed, "variant": variant,
                    "AP": fast_ap(y_true_parent, p_parent_test),
                    "F1_honest": f1_score(y_true_parent, pred, zero_division=0),
                    "threshold": thr_v,
                })

            _record("direct", P[:, PARENT_IDX], P_oof[:, PARENT_IDX])
            for m in PARENT_METHODS:
                _record(
                    m,
                    project_parent(P, CHILD_IDX, PARENT_IDX, m)[:, PARENT_IDX],
                    project_parent(P_oof, CHILD_IDX, PARENT_IDX, m)[:, PARENT_IDX],
                )

            # Hard OR of thresholded children: binary, so only F1 is meaningful.
            y_children = (P[:, CHILD_IDX] >= np.asarray(thr)[CHILD_IDX]).astype(int)
            parent_rows.append({
                "Strategy": strat, "seed": seed, "variant": "hard_or_at_tau",
                "AP": float("nan"),
                "F1_honest": f1_score(y_true_parent, hard_or(y_children), zero_division=0),
                "threshold": float("nan"),
            })

    df_parent = pd.DataFrame(parent_rows)
    df_parent_mean = (df_parent.groupby(["Strategy", "variant"])[["AP", "F1_honest"]]
                      .mean().reset_index())
    piv_ap = df_parent_mean.pivot(index="Strategy", columns="variant", values="AP")
    piv_f1 = df_parent_mean.pivot(index="Strategy", columns="variant", values="F1_honest")
    order = ["direct"] + PARENT_METHODS + ["hard_or_at_tau"]
    piv_ap = piv_ap.reindex(columns=[c for c in order if c in piv_ap.columns])
    piv_f1 = piv_f1.reindex(columns=[c for c in order if c in piv_f1.columns])

    print("\n--- 3a. Antimicrobial AP: predicted directly vs derived from children ---")
    print(piv_ap.round(4).to_string())
    print("\n--- 3b. Antimicrobial honest-F1 ---")
    print(piv_f1.round(4).to_string())

    # ── 4. reverse projection: constrain children by the parent ───────────────
    child_rows = []
    for strat in STRATEGIES:
        for k, seed in enumerate(SEEDS):
            P = test_probs[strat][k]
            base = {"Strategy": strat, "seed": seed, "variant": "direct",
                    "macro_AP_4": _macro_ap(Y_test, P, CHILD_IDX)}
            for j, lbl in zip(CHILD_IDX, ANALYSIS_LABELS):
                base[f"AP_{lbl}"] = fast_ap(Y_test[:, j], P[:, j])
            child_rows.append(base)

            for m in CHILD_METHODS:
                Pp = project_children(P, CHILD_IDX, PARENT_IDX, m)
                rec = {"Strategy": strat, "seed": seed, "variant": m,
                       "macro_AP_4": _macro_ap(Y_test, Pp, CHILD_IDX)}
                for j, lbl in zip(CHILD_IDX, ANALYSIS_LABELS):
                    rec[f"AP_{lbl}"] = fast_ap(Y_test[:, j], Pp[:, j])
                child_rows.append(rec)

    df_child = pd.DataFrame(child_rows)
    df_child_mean = (df_child.groupby(["Strategy", "variant"])
                     .mean(numeric_only=True).drop(columns=["seed"]).reset_index())
    piv_child = df_child_mean.pivot(index="Strategy", columns="variant", values="macro_AP_4")
    piv_child = piv_child.reindex(columns=[c for c in ["direct"] + CHILD_METHODS
                                           if c in piv_child.columns])

    print("\n--- 4. mAP over the 4 activities after constraining children by the parent ---")
    print(piv_child.round(4).to_string())
    print("\n    per-label AP (mean over seeds):")
    per_lbl = df_child_mean.set_index(["Strategy", "variant"])[
        [f"AP_{l}" for l in ANALYSIS_LABELS]
    ]
    print(per_lbl.round(4).to_string())

    # ── 5. paired-bootstrap CIs on the headline deltas ────────────────────────
    print("\n--- 5. Paired-bootstrap CIs on the best derivations ---", flush=True)
    boot_rows = []

    for strat in STRATEGIES:
        direct = [p[:, [PARENT_IDX]] for p in test_probs[strat]]
        for m in PARENT_METHODS:
            derived = [
                project_parent(p, CHILD_IDX, PARENT_IDX, m)[:, [PARENT_IDX]]
                for p in test_probs[strat]
            ]
            r = compute_paired_bootstrap_delta(
                Y_test[:, [PARENT_IDX]], direct, derived,
                n_bootstraps=args.n_bootstraps, seed=args.seed, label_index=0,
            )
            boot_rows.append({
                "Strategy": strat, "target": OR_PARENT,
                "comparison": f"{m} - direct", **r,
            })

    for strat in STRATEGIES:
        direct = [p[:, CHILD_IDX] for p in test_probs[strat]]
        for m in CHILD_METHODS:
            proj = [
                project_children(p, CHILD_IDX, PARENT_IDX, m)[:, CHILD_IDX]
                for p in test_probs[strat]
            ]
            r = compute_paired_bootstrap_delta(
                Y_test[:, CHILD_IDX], direct, proj,
                n_bootstraps=args.n_bootstraps, seed=args.seed,
            )
            boot_rows.append({
                "Strategy": strat, "target": "mAP-4 (children)",
                "comparison": f"{m} - direct", **r,
            })

    df_boot = pd.DataFrame(boot_rows)
    show = df_boot[["Strategy", "target", "comparison", "delta",
                    "ci_lower", "ci_upper", "significant"]].copy()
    for c in ["delta", "ci_lower", "ci_upper"]:
        show[c] = show[c].map(lambda v: f"{v:+.4f}")
    print(show.to_string(index=False))

    # ── save ──────────────────────────────────────────────────────────────────
    df_audit.to_csv(OUT_DIR / "consistency_audit_per_seed.csv", index=False)
    df_audit_mean.round(6).to_csv(OUT_DIR / "consistency_audit_mean.csv")
    df_parent.to_csv(OUT_DIR / "derived_parent_per_seed.csv", index=False)
    piv_ap.round(6).to_csv(OUT_DIR / "derived_parent_ap.csv")
    piv_f1.round(6).to_csv(OUT_DIR / "derived_parent_f1.csv")
    df_child.to_csv(OUT_DIR / "child_projection_per_seed.csv", index=False)
    df_child_mean.round(6).to_csv(OUT_DIR / "child_projection_mean.csv", index=False)
    df_boot.to_csv(OUT_DIR / "orgate_paired_bootstrap.csv", index=False)

    (OUT_DIR / "orgate_summary.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(),
        "generated_by": "scripts/orgate_analysis.py",
        "preds_cache": "data/preds_prob/ (see MANIFEST.json)",
        "parent_label": OR_PARENT,
        "child_labels": ANALYSIS_LABELS,
        "seeds": SEEDS,
        "n_bootstraps": args.n_bootstraps,
        "ground_truth_or_gate": gt,
        "consistency_audit_mean": df_audit_mean.to_dict(),
        "derived_parent_ap": piv_ap.to_dict(),
        "derived_parent_f1": piv_f1.to_dict(),
        "child_projection_macro_ap4": piv_child.to_dict(),
    }, indent=2, default=float))

    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    with RunLogger(
        script_name="orgate_analysis.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=30.0,
        extra_meta={"argv": sys.argv},
    ):
        main()
