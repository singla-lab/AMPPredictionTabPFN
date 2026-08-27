"""
Unit tests for the OR-gate utilities (src/utils/orgate.py).
"""
import numpy as np
import pytest

from src.utils.orgate import (
    hard_or,
    hard_violations,
    max_or,
    noisy_or,
    probability_violations,
    project_children,
    project_parent,
    verify_or_gate,
)


# ── derivations ───────────────────────────────────────────────────────────────

def test_noisy_or_known_values():
    p = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 0.3]])
    got = noisy_or(p)
    assert got[0] == pytest.approx(0.75)     # 1 - 0.5*0.5
    assert got[1] == pytest.approx(0.0)
    assert got[2] == pytest.approx(1.0)


def test_max_or_known_values():
    p = np.array([[0.2, 0.9, 0.4], [0.0, 0.0, 0.0]])
    assert max_or(p).tolist() == pytest.approx([0.9, 0.0])


def test_noisy_or_is_at_least_max_or():
    """1 - Prod(1-p) >= max(p) always, so noisy-OR never undershoots the bound."""
    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, size=(500, 4))
    assert np.all(noisy_or(p) >= max_or(p) - 1e-6)


def test_hard_or():
    y = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert hard_or(y).tolist() == [0, 1, 1, 1]


# ── ground-truth verification ─────────────────────────────────────────────────

def test_verify_or_gate_detects_exact_relation():
    Y = np.array([
        [0, 0, 0],   # children 0,1 ; parent col 2
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ])
    r = verify_or_gate(Y, child_idx=[0, 1], parent_idx=2)
    assert r["holds_exactly"] is True
    assert r["child_without_parent"] == 0
    assert r["parent_without_child"] == 0


def test_verify_or_gate_counts_both_violation_directions():
    Y = np.array([
        [1, 0, 0],   # child active, parent inactive
        [0, 0, 1],   # parent active, no child
    ])
    r = verify_or_gate(Y, child_idx=[0, 1], parent_idx=2)
    assert r["holds_exactly"] is False
    assert r["child_without_parent"] == 1
    assert r["parent_without_child"] == 1


# ── consistency audit ─────────────────────────────────────────────────────────

def test_probability_violations_flags_incoherent_rows():
    # row 0 coherent (parent >= max child); row 1 violates by 0.4
    probs = np.array([
        [0.2, 0.3, 0.9],
        [0.8, 0.1, 0.4],
    ])
    r = probability_violations(probs, child_idx=[0, 1], parent_idx=2)
    assert r["prob_violation_count"] == 1
    assert r["prob_violation_rate"] == pytest.approx(0.5)
    assert r["mean_violation_margin"] == pytest.approx(0.4)
    assert r["max_violation_margin"] == pytest.approx(0.4)


def test_probability_violations_none_when_coherent():
    probs = np.array([[0.1, 0.2, 0.5], [0.0, 0.0, 0.0]])
    r = probability_violations(probs, child_idx=[0, 1], parent_idx=2)
    assert r["prob_violation_count"] == 0
    assert r["mean_violation_margin"] == 0.0


def test_hard_violations_orphan_and_empty():
    probs = np.array([
        [0.9, 0.1, 0.1],   # child fires, parent does not -> orphan_child
        [0.1, 0.1, 0.9],   # parent fires, no child       -> empty_parent
        [0.9, 0.1, 0.9],   # consistent
        [0.1, 0.1, 0.1],   # consistent (all off)
    ])
    r = hard_violations(probs, thresholds=[0.5, 0.5, 0.5],
                        child_idx=[0, 1], parent_idx=2)
    assert r["orphan_child_count"] == 1
    assert r["empty_parent_count"] == 1
    assert r["any_violation_count"] == 2
    assert r["any_violation_rate"] == pytest.approx(0.5)


# ── projections ───────────────────────────────────────────────────────────────

def test_project_parent_replaces_only_parent_column():
    probs = np.array([[0.2, 0.6, 0.05]])
    out = project_parent(probs, child_idx=[0, 1], parent_idx=2, method="max")
    assert out[0, 0] == pytest.approx(0.2)      # children untouched
    assert out[0, 1] == pytest.approx(0.6)
    assert out[0, 2] == pytest.approx(0.6)      # parent := max(children)
    assert probs[0, 2] == pytest.approx(0.05)   # input not mutated


def test_project_parent_methods_and_unknown():
    probs = np.array([[0.5, 0.5, 0.0]])
    assert project_parent(probs, [0, 1], 2, "noisy_or")[0, 2] == pytest.approx(0.75)
    assert project_parent(probs, [0, 1], 2, "mean_max_noisyor")[0, 2] == pytest.approx(0.625)
    with pytest.raises(ValueError, match="Unknown parent derivation"):
        project_parent(probs, [0, 1], 2, "nope")


def test_project_children_clamp_only_touches_violating_rows():
    probs = np.array([
        [0.8, 0.1, 0.4],   # child 0 exceeds parent -> clamped to 0.4
        [0.2, 0.1, 0.9],   # already coherent       -> unchanged
    ])
    out = project_children(probs, child_idx=[0, 1], parent_idx=2, method="clamp")
    assert out[0, 0] == pytest.approx(0.4)
    assert out[0, 1] == pytest.approx(0.1)
    assert out[1, 0] == pytest.approx(0.2)
    assert out[1, 1] == pytest.approx(0.1)
    # After clamping, no probability violation can remain.
    assert probability_violations(out, [0, 1], 2)["prob_violation_count"] == 0


def test_project_children_multiply_rescales_every_row():
    probs = np.array([[0.4, 0.2, 0.5]])
    out = project_children(probs, child_idx=[0, 1], parent_idx=2, method="multiply")
    assert out[0, 0] == pytest.approx(0.2)      # 0.4 * 0.5
    assert out[0, 1] == pytest.approx(0.1)      # 0.2 * 0.5
    assert out[0, 2] == pytest.approx(0.5)      # parent unchanged


def test_project_children_unknown_method():
    with pytest.raises(ValueError, match="Unknown child projection"):
        project_children(np.zeros((2, 3)), [0, 1], 2, "nope")


def test_clamp_is_noop_for_already_coherent_output():
    """
    Label Powerset is coherent by construction: its parent marginal sums every
    powerset class in which any child is active, so p_parent >= p_child holds
    identically. Clamping such output must change nothing.
    """
    rng = np.random.RandomState(1)
    children = rng.uniform(0, 0.4, size=(200, 3))
    parent = children.max(axis=1) + rng.uniform(0.01, 0.3, size=200)
    probs = np.column_stack([children, np.clip(parent, 0, 1)])
    out = project_children(probs, child_idx=[0, 1, 2], parent_idx=3, method="clamp")
    assert np.allclose(out, probs)


# ── prioritisation utilities (src/utils/prioritization.py) ────────────────────

def test_condition_joint_recovers_marginals_and_revealed_column():
    from src.utils.prioritization import condition_joint, combo_matrix
    rng = np.random.RandomState(0)
    L = 3
    joint = rng.dirichlet(np.ones(2 ** L), size=50)
    m = condition_joint(joint, revealed_idx=1, n_labels=L, revealed_value=1)
    assert m.shape == (50, L)
    # The revealed column must be exactly 1 after conditioning on it being 1.
    assert np.allclose(m[:, 1], 1.0)
    assert np.all((m >= -1e-12) & (m <= 1 + 1e-12))


def test_condition_joint_matches_brute_force():
    from src.utils.prioritization import condition_joint, combo_matrix
    rng = np.random.RandomState(1)
    L = 3
    joint = rng.dirichlet(np.ones(2 ** L), size=5)
    combos = combo_matrix(L)
    got = condition_joint(joint, revealed_idx=0, n_labels=L, revealed_value=1)
    for r in range(5):
        keep = combos[:, 0] == 1
        p = joint[r, keep] / joint[r, keep].sum()
        for j in range(L):
            exp = (p * combos[keep, j]).sum()
            assert abs(got[r, j] - exp) < 1e-12


def test_condition_joint_handles_zero_mass_row():
    from src.utils.prioritization import condition_joint, combo_matrix
    L = 2
    joint = np.zeros((1, 4))
    combos = combo_matrix(L)
    joint[0, np.flatnonzero(combos[:, 0] == 0)[0]] = 1.0   # all mass where label0=0
    m = condition_joint(joint, revealed_idx=0, n_labels=L, revealed_value=1)
    assert np.isfinite(m).all()
    assert np.allclose(m[:, 0], 1.0)


def test_build_scenarios_counts():
    from src.utils.prioritization import build_scenarios
    Y = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    sc = build_scenarios(Y)
    # one scenario per (peptide, positive activity): 1 + 2 + 0 = 3
    assert len(sc) == 3
    assert all(len(s["candidates"]) == 2 for s in sc)
    assert all(s["revealed"] not in s["candidates"] for s in sc)


def test_hit_at_k_and_precision_at_k():
    from src.utils.prioritization import hit_at_k, precision_at_k
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])
    truth = np.array([[1, 0, 0], [0, 0, 1]])
    # row0 top1 = col0 (positive) -> hit; row1 top1 = col1 (negative) -> miss
    assert hit_at_k(scores, truth, 1) == 0.5
    # row1 top2 = cols 1,2 and col2 is positive -> both rows now hit
    assert hit_at_k(scores, truth, 2) == 1.0
    assert precision_at_k(scores.ravel(), truth.ravel().astype(float), 2) == 0.5


def test_evaluate_ranker_perfect_vs_random_ordering():
    from src.utils.prioritization import evaluate_ranker
    rng = np.random.RandomState(2)
    truth = (rng.rand(400, 3) < 0.2).astype(int)
    perfect = truth + rng.rand(400, 3) * 1e-6          # positives always rank first
    res = evaluate_ranker(perfect, truth, ks_hit=(1,), ks_prec=(100,))
    assert res["hit@1"] == pytest.approx((truth.sum(axis=1) > 0).mean())
    assert res["prec@100"] == 1.0
    assert res["enrich@100"] > 1.0


def test_bootstrap_hit_delta_detects_better_ranker():
    from src.utils.prioritization import bootstrap_hit_delta
    rng = np.random.RandomState(3)
    truth = (rng.rand(600, 3) < 0.3).astype(int)
    good = truth + rng.rand(600, 3) * 0.5
    bad = rng.rand(600, 3)
    r = bootstrap_hit_delta(bad, good, truth, k=1, n_bootstraps=200)
    assert r["delta"] > 0 and r["ci_lower"] > 0 and r["significant"] is True
