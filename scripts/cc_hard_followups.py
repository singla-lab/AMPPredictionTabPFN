"""
Follow-ups made possible by the CC context ablation, at zero GPU cost.

scripts/cc_context_ablation.py showed that CC's deficit versus BR is an
inference artifact: propagating hard 0/1 context instead of raw probabilities
moves CC by +0.0225 mAP-5 using the *same* fitted chains. Three earlier results
were computed from the soft-context predictions and are therefore contaminated:

  1. Oracle-ceiling capture. results/label_dependence/oracle reports CC as
     capturing -40.8% of the dependence headroom, i.e. actively destroying it.
     The oracle probabilities are unaffected, so recomputing capture against
     cc_hard needs no refitting.
  2. Calibration. CC had the worst calibration of any strategy
     (Antimicrobial ECE 0.057 vs ~0.005 elsewhere).
  3. OR-gate coherence. CC violated p_parent >= max(p_child) on 44.9% of rows
     with margins up to 0.65, by far the worst.

If the soft context is the common cause, all three should improve together.

Reads the cc_soft / cc_hard test probabilities saved by the ablation, plus the
cached oracle probabilities. Writes results/analysis_v2/cc_hard_followups/ —
new files only.

Note: the ablation produced test predictions only, not OOF predictions, so
honest-threshold quantities (honest-F1, decision-level violation rates) cannot
be recomputed for cc_hard here. Those need an OOF run and are reported as
unavailable rather than approximated with a 0.5 cut.

Usage:
    python scripts/cc_hard_followups.py
"""
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
    load_test_probs,
)
from src.utils.metrics import (
    compute_calibration_metrics,
    compute_paired_bootstrap_delta,
    fast_ap,
)
from src.utils.orgate import probability_violations

CC_PREDS = REPO_ROOT / "results" / "analysis_v2" / "cc_context" / "preds"
ORACLE_DIR = REPO_ROOT / "results" / "label_dependence" / "oracle"
OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "cc_hard_followups"

CHILD_IDX = label_indices(ANALYSIS_LABELS)
PARENT_IDX = LABEL_COLS.index(OR_PARENT)


def load_cc_variant(mode: str):
    """Test probabilities for cc_soft / cc_hard, one (n_test, 5) array per seed."""
    out = []
    for seed in SEEDS:
        p = CC_PREDS / f"cc_{mode}_seed{seed}.npy"
        out.append(np.asarray(np.load(p, allow_pickle=True).item()["test_probs"],
                              dtype=np.float32))
    return out


def load_oracle():
    """Oracle test probabilities over the 4 analysis labels, one per seed."""
    return [
        np.asarray(np.load(ORACLE_DIR / f"oracle_seed{s}.npy",
                           allow_pickle=True).item()["test_probs"], dtype=np.float32)
        for s in SEEDS
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, Y_test = load_labels()

    cached = load_test_probs(STRATEGIES, SEEDS)
    variants = {"CC_soft": load_cc_variant("soft"), "CC_hard": load_cc_variant("hard")}
    allp = {**cached, **variants}
    oracle = load_oracle()

    # ── 1. oracle-ceiling capture, recomputed ─────────────────────────────────
    # Capture % = (strategy - BR) / (oracle - BR), on the 4 real activities.
    ap4 = {}
    for name, plist in allp.items():
        per_seed = [
            np.mean([fast_ap(Y_test[:, j], p[:, j]) for j in CHILD_IDX])
            for p in plist
        ]
        ap4[name] = float(np.mean(per_seed))
    ap4_oracle = float(np.mean([
        np.mean([fast_ap(Y_test[:, j], o[:, k]) for k, j in enumerate(CHILD_IDX)])
        for o in oracle
    ]))

    br = ap4["BR"]
    headroom = ap4_oracle - br
    rows = []
    for name in ["BR", "LP", "CC", "CC_soft", "CC_hard", "ECC", "PCC"]:
        rows.append({
            "Strategy": name,
            "mAP4": ap4[name],
            "delta_vs_BR": ap4[name] - br,
            "pct_of_ceiling_captured": 100.0 * (ap4[name] - br) / headroom,
        })
    df_oracle = pd.DataFrame(rows)

    print("=" * 76)
    print("  1. ORACLE-CEILING CAPTURE, RECOMPUTED WITH HARD-CONTEXT CC")
    print("=" * 76)
    print(f"  BR floor (mAP-4)   : {br:.4f}")
    print(f"  Oracle ceiling     : {ap4_oracle:.4f}")
    print(f"  Available headroom : {headroom:+.4f}\n")
    print(df_oracle.round(4).to_string(index=False))

    # ── 2. calibration ────────────────────────────────────────────────────────
    cal_rows = []
    for name, plist in allp.items():
        per_seed = [compute_calibration_metrics(Y_test, p) for p in plist]
        rec = {"Strategy": name,
               "macro_ECE": float(np.mean([c["macro_ece"] for c in per_seed])),
               "macro_Brier": float(np.mean([c["macro_brier"] for c in per_seed]))}
        for j, lbl in enumerate(LABEL_COLS):
            rec[f"ECE_{lbl}"] = float(np.mean([c["ece_per_label"][j] for c in per_seed]))
        cal_rows.append(rec)
    df_cal = pd.DataFrame(cal_rows)

    print("\n" + "=" * 76)
    print("  2. CALIBRATION")
    print("=" * 76)
    print(df_cal.round(4).to_string(index=False))

    # ── 3. OR-gate coherence ──────────────────────────────────────────────────
    coh_rows = []
    for name, plist in allp.items():
        per_seed = [probability_violations(p, CHILD_IDX, PARENT_IDX) for p in plist]
        coh_rows.append({
            "Strategy": name,
            "prob_violation_rate": float(np.mean([v["prob_violation_rate"] for v in per_seed])),
            "mean_violation_margin": float(np.mean([v["mean_violation_margin"] for v in per_seed])),
            "max_violation_margin": float(np.mean([v["max_violation_margin"] for v in per_seed])),
        })
    df_coh = pd.DataFrame(coh_rows)

    print("\n" + "=" * 76)
    print("  3. OR-GATE COHERENCE  (p_parent >= max p_child)")
    print("=" * 76)
    print(df_coh.round(4).to_string(index=False))

    # ── 4. is hard-context CC now competitive with LP/PCC? ────────────────────
    print("\n" + "=" * 76)
    print("  4. PAIRED BOOTSTRAP: CC_hard vs the leaders (mAP-4)")
    print("=" * 76, flush=True)
    boot = []
    for ref in ["LP", "PCC", "ECC"]:
        r = compute_paired_bootstrap_delta(
            Y_test[:, CHILD_IDX],
            [p[:, CHILD_IDX] for p in allp[ref]],
            [p[:, CHILD_IDX] for p in allp["CC_hard"]],
            n_bootstraps=2000,
        )
        boot.append({"comparison": f"CC_hard - {ref}", **r})
    df_boot = pd.DataFrame(boot)
    show = df_boot[["comparison", "delta", "ci_lower", "ci_upper", "significant"]].copy()
    for c in ["delta", "ci_lower", "ci_upper"]:
        show[c] = show[c].map(lambda v: f"{v:+.4f}")
    print(show.to_string(index=False))

    df_oracle.to_csv(OUT_DIR / "oracle_capture_recomputed.csv", index=False)
    df_cal.to_csv(OUT_DIR / "calibration_recomputed.csv", index=False)
    df_coh.to_csv(OUT_DIR / "coherence_recomputed.csv", index=False)
    df_boot.to_csv(OUT_DIR / "cc_hard_vs_leaders_bootstrap.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(),
        "generated_by": "scripts/cc_hard_followups.py",
        "note": ("Recomputes oracle capture, calibration and OR-gate coherence "
                 "using the hard-context CC predictions from the ablation. "
                 "Honest-threshold quantities are unavailable for cc_hard "
                 "because the ablation produced test predictions only."),
        "br_floor_mAP4": br,
        "oracle_ceiling_mAP4": ap4_oracle,
        "headroom": headroom,
        "oracle_capture": df_oracle.to_dict(orient="records"),
    }, indent=2, default=float))
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
