"""
Unit tests for the metrics module (src/utils/metrics.py).
"""
import pytest
import numpy as np
from src.utils.metrics import (
    compute_ap,
    compute_max_f1,
    compute_honest_f1,
    compute_subset_accuracy,
    compute_calibration_metrics,
    compute_all_metrics,
    compute_bootstrap_ci,
    compute_paired_statistical_tests,
)


@pytest.fixture
def dummy_labels_and_probs():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=(100, 5))
    y_probs = rng.uniform(0, 1, size=(100, 5))
    return y_true, y_probs


def test_compute_ap(dummy_labels_and_probs):
    y_true, y_probs = dummy_labels_and_probs
    aps, macro_ap = compute_ap(y_true, y_probs)
    assert len(aps) == 5
    assert 0.0 <= macro_ap <= 1.0
    for ap in aps:
        assert 0.0 <= ap <= 1.0


def test_compute_max_f1(dummy_labels_and_probs):
    y_true, y_probs = dummy_labels_and_probs
    f1s, macro_f1 = compute_max_f1(y_true, y_probs)
    assert len(f1s) == 5
    assert 0.0 <= macro_f1 <= 1.0
    for f1 in f1s:
        assert 0.0 <= f1 <= 1.0


def test_compute_honest_f1(dummy_labels_and_probs):
    y_true, y_probs = dummy_labels_and_probs
    y_train = y_true[:50]
    y_train_probs = y_probs[:50]
    y_test = y_true[50:]
    y_test_probs = y_probs[50:]

    f1s, macro_f1, thresholds = compute_honest_f1(y_train, y_train_probs, y_test, y_test_probs)
    assert len(f1s) == 5
    assert len(thresholds) == 5
    assert 0.0 <= macro_f1 <= 1.0


def test_compute_subset_accuracy(dummy_labels_and_probs):
    y_true, y_probs = dummy_labels_and_probs
    acc = compute_subset_accuracy(y_true, y_probs)
    assert 0.0 <= acc <= 1.0


def test_compute_calibration_metrics(dummy_labels_and_probs):
    y_true, y_probs = dummy_labels_and_probs
    calib = compute_calibration_metrics(y_true, y_probs, n_bins=15)
    assert 'macro_ece' in calib
    assert 'macro_mce' in calib
    assert 'macro_brier' in calib
    assert 0.0 <= calib['macro_ece'] <= 1.0
    assert 0.0 <= calib['macro_mce'] <= 1.0
    assert 0.0 <= calib['macro_brier'] <= 1.0


def test_compute_all_metrics(dummy_labels_and_probs):
    y_true, y_probs = dummy_labels_and_probs
    res = compute_all_metrics(y_true, y_probs)
    assert 'macro_ap' in res
    assert 'macro_max_f1' in res
    assert 'subset_accuracy' in res
    assert 'per_label' in res
    assert len(res['per_label']) == 5


def test_compute_bootstrap_ci(dummy_labels_and_probs):
    y_true, y_probs = dummy_labels_and_probs
    mean_val, low, high = compute_bootstrap_ci(y_true, y_probs, metric_fn=compute_ap, n_bootstraps=100)
    assert low <= mean_val <= high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_compute_paired_statistical_tests():
    vals_a = [0.80, 0.82, 0.81]
    vals_b = [0.85, 0.87, 0.86]
    res = compute_paired_statistical_tests(vals_a, vals_b)
    assert 'delta_mean' in res
    assert res['delta_mean'] > 0


# ── new statistical machinery (paired bootstrap, fast AP) ─────────────────────

def test_fast_ap_matches_sklearn():
    """fast_ap must reproduce sklearn's average_precision_score."""
    from sklearn.metrics import average_precision_score
    from src.utils.metrics import fast_ap

    rng = np.random.RandomState(0)
    for _ in range(20):
        y = rng.randint(0, 2, size=500)
        p = rng.uniform(0, 1, size=500)
        if len(np.unique(y)) < 2:
            continue
        assert abs(fast_ap(y, p) - average_precision_score(y, p)) < 1e-9


def test_fast_ap_degenerate_column():
    """A single-class column returns 0.5, matching compute_ap's convention."""
    from src.utils.metrics import fast_ap
    assert fast_ap(np.zeros(50, dtype=int), np.random.rand(50)) == 0.5
    assert fast_ap(np.ones(50, dtype=int), np.random.rand(50)) == 0.5


def test_fast_ap_perfect_and_inverted_ranking():
    from src.utils.metrics import fast_ap
    y = np.array([0, 0, 1, 1])
    assert fast_ap(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert fast_ap(y, np.array([0.9, 0.8, 0.2, 0.1])) < 0.6


def test_subset_accuracy_per_label_thresholds():
    """Per-label thresholds must be applied column-wise, not globally."""
    y_true = np.array([[1, 0], [0, 1]])
    y_probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    # A global 0.5 cut gets both rows exactly right.
    assert compute_subset_accuracy(y_true, y_probs, threshold=0.5) == 1.0
    # Raising label 0's cut above 0.6 breaks row 0 only.
    assert compute_subset_accuracy(y_true, y_probs, thresholds=[0.7, 0.5]) == 0.5


def test_paired_bootstrap_delta_detects_real_improvement():
    """B strictly better than A on the same rows -> positive, significant delta."""
    from src.utils.metrics import compute_paired_bootstrap_delta

    rng = np.random.RandomState(3)
    n = 800
    y = rng.randint(0, 2, size=(n, 2))
    # Gaussian noise on the score keeps the classes genuinely overlapping, so
    # AP stays well below 1 and the two arms are actually distinguishable.
    probs_a = y + rng.randn(n, 2) * 2.0      # weakly informative
    probs_b = y + rng.randn(n, 2) * 0.4      # strongly informative

    res = compute_paired_bootstrap_delta(y, probs_a, probs_b, n_bootstraps=200)
    assert res["delta"] > 0
    assert res["ci_lower"] > 0
    assert res["significant"] is True


def test_paired_bootstrap_delta_identical_inputs():
    """Identical predictions -> exactly zero delta and a degenerate interval."""
    from src.utils.metrics import compute_paired_bootstrap_delta

    rng = np.random.RandomState(4)
    y = rng.randint(0, 2, size=(300, 2))
    p = rng.uniform(0, 1, size=(300, 2))
    res = compute_paired_bootstrap_delta(y, p, p, n_bootstraps=100)
    assert res["delta"] == pytest.approx(0.0, abs=1e-12)
    assert res["ci_lower"] == pytest.approx(0.0, abs=1e-12)
    assert res["ci_upper"] == pytest.approx(0.0, abs=1e-12)
    assert res["significant"] is False


def test_pooling_seeds_narrows_ci_vs_correct_method():
    """
    Regression guard for the CI bug: vstacking S seeds into one pool and
    bootstrapping 3N rows yields an interval ~sqrt(S) too narrow versus
    resampling the N real test rows and averaging seeds inside the resample.
    """
    from src.utils.metrics import (
        compute_bootstrap_ci,
        compute_bootstrap_ci_seed_averaged,
    )

    rng = np.random.RandomState(5)
    n, n_seeds = 600, 3
    y = rng.randint(0, 2, size=(n, 3))
    # Overlapping classes so AP < 1 and the bootstrap has real variance to measure.
    probs = [y + rng.randn(n, 3) * 1.5 for _ in range(n_seeds)]

    correct = compute_bootstrap_ci_seed_averaged(y, probs, n_bootstraps=300)
    width_correct = correct["ci_upper"] - correct["ci_lower"]

    y_pool = np.vstack([y] * n_seeds)
    p_pool = np.vstack(probs)
    _, lo, hi = compute_bootstrap_ci(y_pool, p_pool, n_bootstraps=300)
    width_pooled = hi - lo

    assert width_pooled < width_correct
    assert width_correct / width_pooled == pytest.approx(np.sqrt(n_seeds), rel=0.35)


def test_bootstrap_strategy_comparison_shape_and_pairing():
    from src.utils.metrics import bootstrap_strategy_comparison

    rng = np.random.RandomState(6)
    n = 500
    y = rng.randint(0, 2, size=(n, 2))
    base = [y + rng.randn(n, 2) * 2.0 for _ in range(2)]
    better = [y + rng.randn(n, 2) * 0.4 for _ in range(2)]

    res = bootstrap_strategy_comparison(
        y, {"BR": base, "LP": better}, reference="BR",
        n_bootstraps=150, label_names=["a", "b"],
        macro_subsets={"all": ["a", "b"], "a_only": ["a"]},
    )
    assert set(res["per_strategy"]) == {"BR", "LP"}
    assert "BR" not in res["deltas_vs_reference"]      # reference has no self-delta

    d_all = res["deltas_vs_reference"]["LP"]["macro"]["all"]
    assert d_all["delta_AP"] > 0
    assert d_all["ci_lower"] > 0
    assert d_all["significant"] is True

    # Point delta must equal the difference of the two point estimates.
    d = (res["per_strategy"]["LP"]["macro"]["all"]["AP"]
         - res["per_strategy"]["BR"]["macro"]["all"]["AP"])
    assert d_all["delta_AP"] == pytest.approx(d)

    # A single-label subset must reproduce that label's per-label numbers exactly.
    assert (res["per_strategy"]["LP"]["macro"]["a_only"]["AP"]
            == pytest.approx(res["per_strategy"]["LP"]["per_label"]["a"]["AP"]))
    assert (res["deltas_vs_reference"]["LP"]["macro"]["a_only"]["delta_AP"]
            == pytest.approx(res["deltas_vs_reference"]["LP"]["per_label"]["a"]["delta_AP"]))


def test_bootstrap_strategy_comparison_rejects_unknown_subset_label():
    from src.utils.metrics import bootstrap_strategy_comparison

    rng = np.random.RandomState(7)
    y = rng.randint(0, 2, size=(100, 2))
    p = [y + rng.randn(100, 2)]
    with pytest.raises(ValueError, match="unknown labels"):
        bootstrap_strategy_comparison(
            y, {"BR": p}, reference="BR", n_bootstraps=10,
            label_names=["a", "b"], macro_subsets={"bad": ["a", "zzz"]},
        )
