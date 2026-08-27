"""
Next-assay prioritisation under a positive-unlabelled (PU) protocol.

Setting. Activity databases record what was *found* positive; an activity a
peptide was never tested for is unlabelled, not known-negative. So the realistic
question is: given that a peptide is already known positive for one activity,
which of its remaining activities should be assayed next?

Protocol implemented here (m = 1):
  * A scenario is a (peptide, revealed activity) pair where the revealed activity
    is a TRUE positive for that peptide.
  * The other L-1 activities are the candidate assays; their true labels are
    treated as unknown at ranking time and revealed only for scoring.
  * A ranker assigns a score to each candidate. Metrics are Hit@k over the
    candidates within a scenario, and precision@k over all candidate pairs pooled
    across scenarios under a global assay budget.

Joint-model rankers condition the joint posterior on the revealed positive,
renormalise, and marginalise — the operation described in
06_experiments/exp_024/run_pcc.py on the `main` branch:
`P(a_next = 1 | x, seen positives)`.

Combination indexing convention used throughout: combos are
``list(itertools.product([0, 1], repeat=L))``, so index c corresponds to a tuple
whose j-th element is the value of label j, with label 0 varying slowest. This
matches src/models/probabilistic_cc.py. It is NOT the LSB-first bit encoding used
by exp_024 on the `main` branch.
"""
import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np


def combo_matrix(n_labels: int) -> np.ndarray:
    """(2^L, L) int array of all label combinations, in canonical order."""
    return np.array(list(itertools.product([0, 1], repeat=n_labels)), dtype=np.int8)


def condition_joint(
    joint: np.ndarray,
    revealed_idx: int,
    n_labels: int,
    revealed_value: int = 1,
) -> np.ndarray:
    """
    Condition a joint posterior on one revealed label and return the marginals.

    Args:
        joint:        (n, 2^L) row-normalised posterior over label combinations.
        revealed_idx: index of the label whose value is known.
        revealed_value: the known value (1 for a revealed positive).

    Returns
    -------
    (n, L) array of P(y_j = 1 | x, y_revealed = revealed_value). The revealed
    column is exactly `revealed_value` by construction.
    """
    combos = combo_matrix(n_labels)
    keep = combos[:, revealed_idx] == revealed_value
    sub = joint[:, keep]
    denom = sub.sum(axis=1, keepdims=True)
    # A row whose entire consistent mass is zero carries no information; fall
    # back to a uniform posterior over the consistent combinations rather than
    # producing NaN.
    degenerate = denom[:, 0] <= 0
    if degenerate.any():
        sub = sub.copy()
        sub[degenerate] = 1.0 / keep.sum()
        denom[degenerate] = 1.0
    post = sub / denom
    return post @ combos[keep].astype(np.float64)


def condition_joint_multi(
    joint: np.ndarray,
    revealed: Dict[int, int],
    n_labels: int,
) -> np.ndarray:
    """
    Condition a joint posterior on several revealed labels at once.

    Args:
        joint:    (n, 2^L) row-normalised posterior over label combinations.
        revealed: {label index: known value}. An empty dict returns the
                  unconditional marginals.

    Returns
    -------
    (n, L) array of P(y_j = 1 | x, revealed).
    """
    combos = combo_matrix(n_labels)
    keep = np.ones(len(combos), dtype=bool)
    for idx, val in revealed.items():
        keep &= combos[:, idx] == val
    if not keep.any():
        raise ValueError(f"No label combination is consistent with {revealed}")
    sub = joint[:, keep]
    denom = sub.sum(axis=1, keepdims=True)
    degenerate = denom[:, 0] <= 0
    if degenerate.any():
        sub = sub.copy()
        sub[degenerate] = 1.0 / keep.sum()
        denom[degenerate] = 1.0
    return (sub / denom) @ combos[keep].astype(np.float64)


def rank_of_target(scores: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """
    1-based rank of the target among the scored candidates (higher score = better).

    Args:
        scores:     (S, C) candidate scores.
        target_pos: (S,) column index of the target within each row.
    """
    tgt = scores[np.arange(len(scores)), target_pos][:, None]
    # Ties count against the target, so the reported rank is conservative.
    return 1 + (scores > tgt).sum(axis=1)


def build_scenarios(Y: np.ndarray) -> List[dict]:
    """
    Enumerate (peptide, revealed positive) scenarios.

    For every peptide and every activity that is a true positive for it, one
    scenario is produced in which that activity is revealed and the remaining
    L-1 activities are candidates.

    Returns a list of dicts with 'row' (peptide index), 'revealed' (label index)
    and 'candidates' (list of label indices).
    """
    n, L = Y.shape
    out = []
    for k in range(L):
        rows = np.flatnonzero(Y[:, k] == 1)
        cands = [j for j in range(L) if j != k]
        for r in rows:
            out.append({"row": int(r), "revealed": k, "candidates": cands})
    return out


def scenario_arrays(scenarios: List[dict], Y: np.ndarray):
    """
    Flatten scenarios into aligned arrays.

    Returns
    -------
    rows       (S,)   peptide index per scenario
    revealed   (S,)   revealed label index per scenario
    cand_idx   (S, C) candidate label indices
    cand_true  (S, C) true labels of the candidates
    """
    rows = np.array([s["row"] for s in scenarios], dtype=np.int64)
    revealed = np.array([s["revealed"] for s in scenarios], dtype=np.int64)
    cand_idx = np.array([s["candidates"] for s in scenarios], dtype=np.int64)
    cand_true = Y[rows[:, None], cand_idx]
    return rows, revealed, cand_idx, cand_true


def hit_at_k(scores: np.ndarray, truth: np.ndarray, k: int) -> float:
    """
    Fraction of scenarios whose top-k ranked candidates contain a true positive.

    Args:
        scores: (S, C) candidate scores.
        truth:  (S, C) candidate ground truth.

    Ties are broken by candidate order, which is fixed across rankers, so no
    ranker gains from tie ordering.
    """
    order = np.argsort(-scores, axis=1, kind="stable")
    topk = np.take_along_axis(truth, order[:, :k], axis=1)
    return float((topk.sum(axis=1) > 0).mean())


def precision_at_k(scores_flat: np.ndarray, truth_flat: np.ndarray, k: int) -> float:
    """Fraction of the globally top-k candidate assays that are true positives."""
    k = min(k, len(scores_flat))
    idx = np.argsort(-scores_flat, kind="stable")[:k]
    return float(truth_flat[idx].mean())


def average_precision_flat(scores_flat: np.ndarray, truth_flat: np.ndarray) -> float:
    """Average precision over the pooled candidate pool."""
    n_pos = int(truth_flat.sum())
    if n_pos == 0 or n_pos == len(truth_flat):
        return float("nan")
    order = np.argsort(-scores_flat, kind="stable")
    y = truth_flat[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / n_pos)


def evaluate_ranker(
    scores: np.ndarray,
    cand_true: np.ndarray,
    ks_hit: Sequence[int] = (1, 2),
    ks_prec: Sequence[int] = (100, 250, 500, 1000, 2000, 5000),
) -> Dict[str, float]:
    """
    Full metric suite for one ranker.

    Returns Hit@k (within scenario), precision@k and enrichment (pooled across
    all candidate pairs), the pooled base rate, and pooled AP.
    """
    sf = scores.ravel()
    tf = cand_true.ravel().astype(float)
    base = float(tf.mean())
    out: Dict[str, float] = {"base_rate": base, "n_scenarios": int(len(scores)),
                             "n_candidate_pairs": int(len(tf))}
    for k in ks_hit:
        out[f"hit@{k}"] = hit_at_k(scores, cand_true, k)
    for k in ks_prec:
        p = precision_at_k(sf, tf, k)
        out[f"prec@{k}"] = p
        out[f"enrich@{k}"] = p / base if base > 0 else float("nan")
    out["AP_pool"] = average_precision_flat(sf, tf)
    return out


def bootstrap_hit_delta(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cand_true: np.ndarray,
    k: int = 1,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap over scenarios on the Hit@k difference (B - A).

    Both rankers are evaluated on the same resampled scenarios each iteration, so
    the shared scenario-sampling variance cancels.
    """
    S = len(cand_true)
    order_a = np.argsort(-scores_a, axis=1, kind="stable")
    order_b = np.argsort(-scores_b, axis=1, kind="stable")
    hit_a = (np.take_along_axis(cand_true, order_a[:, :k], axis=1).sum(axis=1) > 0)
    hit_b = (np.take_along_axis(cand_true, order_b[:, :k], axis=1).sum(axis=1) > 0)

    delta = float(hit_b.mean() - hit_a.mean())
    rng = np.random.RandomState(seed)
    d = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = rng.randint(0, S, S)
        d[i] = hit_b[idx].mean() - hit_a[idx].mean()
    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.percentile(d, [100 * alpha, 100 * (1 - alpha)])
    return {
        "delta": delta,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "pseudo_p": float(np.mean(d <= 0)),
        "significant": bool(lo > 0 or hi < 0),
        "n_scenarios": int(S),
    }
