"""
OR-gate utilities for the deterministic Antimicrobial parent label.

In this dataset ``Antimicrobial == OR(Antibacterial, Antifungal, Antiviral,
Antiparasitic)`` holds in every row of every split, with zero exceptions. Every
strategy nonetheless predicts Antimicrobial as an ordinary fifth label, which
leaves two things on the table:

  - the parent can be *derived* from the four children rather than predicted,
  - the constraint ``y_child = 1 => y_parent = 1`` can be *enforced*, which in
    probability space means ``p_parent >= max(p_children)``; the contrapositive
    lets a confident parent prediction suppress spurious child predictions.

This module provides the derivations, the consistency audit, and the two
projection directions. All functions are pure NumPy and operate on cached
probability matrices — no model refitting.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np


# ── parent derivations (children -> parent) ───────────────────────────────────

def noisy_or(p_children: np.ndarray) -> np.ndarray:
    """
    P(parent) = 1 - Π_j (1 - p_j).

    The probability that at least one child activity is present, under the
    assumption that the children are conditionally independent given x. That
    assumption is exactly what the dependence analyses show to be imperfect, so
    noisy-OR is expected to overshoot where children co-occur.
    """
    p = np.clip(np.asarray(p_children, dtype=np.float64), 0.0, 1.0)
    return (1.0 - np.prod(1.0 - p, axis=1)).astype(np.float32)


def max_or(p_children: np.ndarray) -> np.ndarray:
    """
    P(parent) = max_j p_j.

    The tightest bound implied by the constraint alone: the parent is at least
    as probable as its most probable child, with no independence assumption.
    """
    return np.max(np.asarray(p_children, dtype=np.float32), axis=1)


def hard_or(y_children: np.ndarray) -> np.ndarray:
    """Binary OR of already-thresholded child predictions."""
    return (np.asarray(y_children).astype(bool).any(axis=1)).astype(int)


# ── projections ───────────────────────────────────────────────────────────────

def project_parent(
    probs: np.ndarray,
    child_idx: Sequence[int],
    parent_idx: int,
    method: str = "max",
) -> np.ndarray:
    """
    Return a copy of `probs` with the parent column replaced by a derivation
    from the children.

    method: 'max' | 'noisy_or' | 'mean_max_noisyor'
    """
    out = np.array(probs, dtype=np.float32, copy=True)
    pc = out[:, list(child_idx)]
    if method == "max":
        out[:, parent_idx] = max_or(pc)
    elif method == "noisy_or":
        out[:, parent_idx] = noisy_or(pc)
    elif method == "mean_max_noisyor":
        out[:, parent_idx] = 0.5 * (max_or(pc) + noisy_or(pc))
    else:
        raise ValueError(f"Unknown parent derivation method: {method!r}")
    return out


def project_children(
    probs: np.ndarray,
    child_idx: Sequence[int],
    parent_idx: int,
    method: str = "clamp",
) -> np.ndarray:
    """
    Return a copy of `probs` with the child columns constrained by the parent.

    Because y_child = 1 implies y_parent = 1, a coherent joint must satisfy
    p_child <= p_parent. Two ways to impose it:

      'clamp'    : p_child <- min(p_child, p_parent). Minimal edit; only touches
                   rows that actually violate the constraint.
      'multiply' : p_child <- p_child * p_parent. Reads the child score as
                   P(child | parent) and reweights by parent confidence, so it
                   rescales every row, not just violating ones.
    """
    out = np.array(probs, dtype=np.float32, copy=True)
    parent = out[:, parent_idx][:, None]
    cols = list(child_idx)
    if method == "clamp":
        out[:, cols] = np.minimum(out[:, cols], parent)
    elif method == "multiply":
        out[:, cols] = out[:, cols] * parent
    else:
        raise ValueError(f"Unknown child projection method: {method!r}")
    return out


# ── consistency audit ─────────────────────────────────────────────────────────

def probability_violations(
    probs: np.ndarray,
    child_idx: Sequence[int],
    parent_idx: int,
    tol: float = 0.0,
) -> Dict[str, float]:
    """
    How badly does a strategy's probability output break p_parent >= max(p_child)?

    Returns the violation rate plus the mean and worst margin over violating
    rows, where margin = max(p_children) - p_parent.
    """
    p = np.asarray(probs, dtype=np.float64)
    gap = p[:, list(child_idx)].max(axis=1) - p[:, parent_idx]
    viol = gap > tol
    return {
        "prob_violation_rate": float(viol.mean()),
        "prob_violation_count": int(viol.sum()),
        "mean_violation_margin": float(gap[viol].mean()) if viol.any() else 0.0,
        "max_violation_margin": float(gap.max()),
    }


def hard_violations(
    probs: np.ndarray,
    thresholds: Sequence[float],
    child_idx: Sequence[int],
    parent_idx: int,
) -> Dict[str, float]:
    """
    Decision-level consistency after thresholding at the given per-label cuts.

    Two failure directions:
      - orphan_child : some child predicted active but parent predicted inactive
      - empty_parent : parent predicted active but no child predicted active
    """
    thr = np.asarray(thresholds, dtype=float).reshape(1, -1)
    y = (np.asarray(probs) >= thr).astype(int)
    child_any = y[:, list(child_idx)].any(axis=1)
    parent = y[:, parent_idx].astype(bool)

    orphan = child_any & ~parent
    empty = parent & ~child_any
    return {
        "orphan_child_rate": float(orphan.mean()),
        "orphan_child_count": int(orphan.sum()),
        "empty_parent_rate": float(empty.mean()),
        "empty_parent_count": int(empty.sum()),
        "any_violation_rate": float((orphan | empty).mean()),
        "any_violation_count": int((orphan | empty).sum()),
    }


def verify_or_gate(Y: np.ndarray, child_idx: Sequence[int], parent_idx: int) -> Dict:
    """
    Confirm the OR relation holds in a ground-truth label matrix.

    Returns the mismatch count in each direction; both must be zero for the
    derived-parent and projection analyses to be licensed.
    """
    Y = np.asarray(Y)
    child_any = Y[:, list(child_idx)].any(axis=1)
    parent = Y[:, parent_idx].astype(bool)
    return {
        "n_rows": int(len(Y)),
        "child_without_parent": int((child_any & ~parent).sum()),
        "parent_without_child": int((parent & ~child_any).sum()),
        "holds_exactly": bool(np.array_equal(child_any, parent)),
    }
