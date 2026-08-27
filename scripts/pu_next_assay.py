"""
T7 — PU next-assay ranking: hold out one positive, reveal m others, rank the rest.

Protocol (as specified in ANALYSIS_AND_NEXT_RUNS.md T7):

  1. For each test peptide with at least one confirmed positive activity, hold out
     one positive activity as the TARGET.
  2. Reveal the m highest-prevalence remaining positives as OBSERVED (m = 0, 1, 2).
  3. Score every unobserved activity (the target included) with a conditional
     posterior P(a = 1 | x, observed positives), obtained by slicing the cached
     joint, renormalising and marginalising.
  4. Metrics: Hit@1, MRR, pooled PU average precision.
  5. Permuted control: replace the peptide's observed activity with a positive
     activity sampled from a DIFFERENT peptide. Same amount of conditioning
     information, link to this specific peptide severed.

Target selection. The spec pairs "hold out one positive" with "reveal the m
highest-prevalence remaining", so the target is the peptide's LOWEST-prevalence
positive and the revealed set is the m highest-prevalence of those remaining.
Global prevalence order (train): Antibacterial 19.61% > Antifungal 8.10% >
Antiviral 5.74% > Antiparasitic 0.52%. One scenario per peptide, which
reproduces the eligible counts in the spec (4235 / 1035 / 250).

Reading the m-progression. Hit@1 necessarily rises with m because the candidate
set shrinks (4 -> 3 -> 2, chance 0.25 -> 0.333 -> 0.50) AND the eligible subset
changes (peptides with more positives are easier). Neither effect is model
quality. The interpretable quantity is the margin over the permuted control at
FIXED m. Chance level, eligible n, candidate-set size and the permuted control
are reported for every m; no raw Hit@1 appears without them.

Rankers:
  BR         cached Binary Relevance marginals; ignores the observed labels.
  LP-joint   LP powerset posterior conditioned on the observed positives.
  PCC-joint  PCC exact 2^L joint conditioned on the observed positives.
  *-perm     the same joint conditioned on a donor peptide's positive instead.

Writes to results/analysis_v2/pu_next_assay/. Creates new files; overwrites nothing.

Usage:
    python scripts/pu_next_assay.py
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

from src.data.preds_cache import (
    ANALYSIS_LABELS,
    SEEDS,
    label_indices,
    load_labels,
    load_test_probs,
)
from src.utils import RunLogger
from src.utils.prioritization import (
    average_precision_flat,
    condition_joint_multi,
    rank_of_target,
)

OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "pu_next_assay"
JOINT_DIR = REPO_ROOT / "results" / "analysis_v2" / "prioritization" / "joints"

CHILD_IDX = label_indices(ANALYSIS_LABELS)
L = len(ANALYSIS_LABELS)
PREVALENCE_ORDER = [0, 1, 2, 3]   # AB > AF > AV > AP, descending train prevalence


def build_scenarios(Y, m):
    """One scenario per eligible peptide (>= m+1 positives)."""
    rows, targets, observed, unobserved = [], [], [], []
    for r in range(len(Y)):
        pos = [j for j in PREVALENCE_ORDER if Y[r, j] == 1]
        if len(pos) < m + 1:
            continue
        target = pos[-1]                        # lowest-prevalence positive
        rest = [j for j in pos if j != target]  # prevalence-ordered already
        obs = rest[:m]
        unobs = [j for j in range(L) if j not in obs]
        rows.append(r); targets.append(target)
        observed.append(obs); unobserved.append(unobs)
    return (np.array(rows), np.array(targets),
            np.array(observed, dtype=np.int64).reshape(len(rows), m),
            np.array(unobserved, dtype=np.int64))


def permuted_observed(Y, rows, targets, m, rng):
    """
    Donor control: draw the observed labels from a different peptide's positives.
    Resampled so the donor set excludes the target, keeping the unobserved set
    the same size and the chance level directly comparable.
    """
    if m == 0:
        return np.zeros((len(rows), 0), dtype=np.int64)
    donors = [r for r in range(len(Y)) if Y[r].sum() >= m]
    out = np.empty((len(rows), m), dtype=np.int64)
    for i, (r, tgt) in enumerate(zip(rows, targets)):
        for _ in range(200):
            d = donors[rng.randint(len(donors))]
            if d == r:
                continue
            dpos = [j for j in PREVALENCE_ORDER if Y[d, j] == 1 and j != tgt]
            if len(dpos) >= m:
                out[i] = dpos[:m]
                break
        else:
            out[i] = [j for j in range(L) if j != tgt][:m]
    return out


def score_scenarios(joint, rows, observed, unobserved):
    """Conditional posterior per scenario, restricted to the unobserved labels."""
    S, C = unobserved.shape
    scores = np.empty((S, C), dtype=np.float64)
    key = [tuple(o) for o in observed]
    for k in sorted(set(key)):
        sel = np.array([i for i, kk in enumerate(key) if kk == k])
        marg = condition_joint_multi(joint[rows[sel]].astype(np.float64),
                                     {int(j): 1 for j in k}, L)
        scores[sel] = np.take_along_axis(marg, unobserved[sel], axis=1)
    return scores


def metrics(scores, unobserved, targets, Y, rows):
    tpos = np.array([list(u).index(t) for u, t in zip(unobserved, targets)])
    rank = rank_of_target(scores, tpos)
    truth = Y[rows[:, None], unobserved].astype(float)
    return {
        "hit@1": float((rank == 1).mean()),
        "MRR": float((1.0 / rank).mean()),
        "PU_AP": average_precision_flat(scores.ravel(), truth.ravel()),
        "mean_rank": float(rank.mean()),
    }, rank


def boot_ci(vals, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    v = np.asarray(vals, dtype=float)
    b = np.array([v[rng.randint(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_bootstraps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _, Y5 = load_labels()
    Y = Y5[:, CHILD_IDX]
    br = [p[:, CHILD_IDX] for p in load_test_probs(["BR"], args.seeds)["BR"]]
    joints = {k: [np.load(JOINT_DIR / f"{k}_joint_seed{s}.npy") for s in args.seeds]
              for k in ("lp", "pcc")
              if all((JOINT_DIR / f"{k}_joint_seed{s}.npy").exists() for s in args.seeds)}
    print(f"joints available: {list(joints)}", flush=True)

    rows_out, rank_store = [], {}
    for m in (0, 1, 2):
        rows, targets, observed, unobserved = build_scenarios(Y, m)
        S, C = unobserved.shape
        chance = 1.0 / C
        print(f"\n{'='*80}\n  m = {m}   eligible peptides = {S}   "
              f"unobserved candidates = {C}   chance Hit@1 = {chance:.4f}\n{'='*80}",
              flush=True)

        rng = np.random.RandomState(1234 + m)
        perm_obs = permuted_observed(Y, rows, targets, m, rng)

        rankers = {"BR": [np.take_along_axis(p[rows], unobserved, axis=1) for p in br]}
        for k, js in joints.items():
            rankers[f"{k.upper()}-joint"] = [
                score_scenarios(j, rows, observed, unobserved) for j in js]
            if m > 0:
                rankers[f"{k.upper()}-perm"] = [
                    score_scenarios(j, rows, perm_obs, unobserved) for j in js]

        for name, mats in rankers.items():
            per_seed, ranks = [], []
            for sc in mats:
                mt, rk = metrics(sc, unobserved, targets, Y, rows)
                per_seed.append(mt); ranks.append(rk)
            hit_vec = np.mean([(r == 1).astype(float) for r in ranks], axis=0)
            lo, hi = boot_ci(hit_vec, args.n_bootstraps)
            rows_out.append({
                "m": m, "ranker": name, "n_eligible": S, "n_candidates": C,
                "chance_hit@1": chance,
                "hit@1": float(np.mean([p["hit@1"] for p in per_seed])),
                "hit@1_ci_lower": lo, "hit@1_ci_upper": hi,
                "MRR": float(np.mean([p["MRR"] for p in per_seed])),
                "PU_AP": float(np.mean([p["PU_AP"] for p in per_seed])),
                "mean_rank": float(np.mean([p["mean_rank"] for p in per_seed])),
                "hit@1_per_seed": [round(p["hit@1"], 4) for p in per_seed]})
            rank_store[(m, name)] = hit_vec

        df = pd.DataFrame([r for r in rows_out if r["m"] == m])
        print(df[["ranker", "hit@1", "hit@1_ci_lower", "hit@1_ci_upper",
                  "MRR", "PU_AP", "mean_rank"]].round(4).to_string(index=False))
        print(f"  chance Hit@1 = {chance:.4f}   n = {S}   candidates = {C}")

        for k in joints:
            real, perm = f"{k.upper()}-joint", f"{k.upper()}-perm"
            if (m, perm) in rank_store:
                d = rank_store[(m, real)] - rank_store[(m, perm)]
                lo, hi = boot_ci(d, args.n_bootstraps)
                print(f"  MARGIN OVER PERMUTED CONTROL [{k.upper()}]: {d.mean():+.4f}"
                      f"  95% CI [{lo:+.4f}, {hi:+.4f}]"
                      f"  {'significant' if lo > 0 or hi < 0 else 'ns'}")

    pd.DataFrame(rows_out).to_csv(OUT_DIR / "pu_next_assay_summary.csv", index=False)

    tb = []
    for m in (0, 1, 2):
        rows, targets, observed, unobserved = build_scenarios(Y, m)
        C = unobserved.shape[1]
        for k, js in joints.items():
            mats = [score_scenarios(j, rows, observed, unobserved) for j in js]
            for t, lab in enumerate(ANALYSIS_LABELS):
                sel = targets == t
                if sel.sum() < 5:
                    continue
                tpos = np.array([list(u).index(tt) for u, tt in
                                 zip(unobserved[sel], targets[sel])])
                hs = [float((rank_of_target(sc[sel], tpos) == 1).mean()) for sc in mats]
                tb.append({"m": m, "ranker": f"{k.upper()}-joint", "target": lab,
                           "n": int(sel.sum()), "chance": 1.0 / C,
                           "hit@1": float(np.mean(hs))})
    df_t = pd.DataFrame(tb)
    if len(df_t):
        df_t.to_csv(OUT_DIR / "pu_next_assay_by_target.csv", index=False)
        print("\n" + "=" * 80)
        print("  Hit@1 by TARGET activity")
        print("=" * 80)
        prim = "PCC-joint" if "pcc" in joints else "LP-joint"
        print(df_t[df_t.ranker == prim][["m", "target", "n", "chance", "hit@1"]]
              .round(4).to_string(index=False))

    (OUT_DIR / "pu_next_assay_summary.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(),
        "generated_by": "scripts/pu_next_assay.py",
        "protocol": ("hold out lowest-prevalence positive as target; reveal m "
                     "highest-prevalence remaining positives; rank all unobserved"),
        "labels": ANALYSIS_LABELS, "seeds": args.seeds,
        "rankers": sorted({r["ranker"] for r in rows_out}),
        "results": rows_out,
    }, indent=2, default=float))
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    with RunLogger(script_name="pu_next_assay.py", log_dir=REPO_ROOT / "logs",
                   sample_interval_s=30.0, extra_meta={"argv": sys.argv}):
        main()
