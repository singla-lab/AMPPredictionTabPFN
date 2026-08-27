"""
Unit tests for TabPFN multilabel classification strategies.
Tests BR, LP, CC, ECC and PCC on a small synthetic dataset (CPU).
"""
import numpy as np
import pytest

from src.data.loader import make_synthetic_data
from src.models import (
    BinaryRelevance,
    ClassifierChain,
    EnsembleClassifierChain,
    LabelPowerset,
    ProbabilisticClassifierChain,
)


@pytest.fixture(scope="module")
def synthetic_data():
    return make_synthetic_data(n_samples=60, n_features=10, n_labels=5, seed=42)


@pytest.mark.parametrize("model_cls, kwargs", [
    (BinaryRelevance, {"n_estimators": 1, "device": "cpu"}),
    (LabelPowerset, {"n_estimators": 1, "device": "cpu"}),
    (ClassifierChain, {"n_estimators": 1, "device": "cpu"}),
    (EnsembleClassifierChain, {"n_estimators": 1, "n_chains": 2, "device": "cpu"}),
    (ProbabilisticClassifierChain, {"n_estimators": 1, "device": "cpu"}),
])
def test_strategy_fit_predict(synthetic_data, model_cls, kwargs):
    X_train, Y_train, X_test, Y_test = synthetic_data

    model = model_cls(**kwargs)
    model.fit(X_train, Y_train)

    probs = model.predict_proba(X_test)
    assert probs.shape == (len(X_test), 5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = model.predict(X_test, threshold=0.5)
    assert preds.shape == (len(X_test), 5)
    assert set(np.unique(preds)).issubset({0, 1})


def test_classifier_chain_context_modes_differ(synthetic_data):
    """
    A chain fits on ground-truth binary context but can propagate either the
    raw probability (probabilistic=True) or a hard 0/1 (probabilistic=False)
    at inference. The two modes are genuinely different inference procedures,
    which is what scripts/cc_context_ablation.py measures.
    """
    X_train, Y_train, X_test, _ = synthetic_data

    soft = ClassifierChain(n_estimators=1, device="cpu", probabilistic=True, seed=0)
    hard = ClassifierChain(n_estimators=1, device="cpu", probabilistic=False, seed=0)
    soft.fit(X_train, Y_train)
    hard.fit(X_train, Y_train)

    p_soft = soft.predict_proba(X_test)
    p_hard = hard.predict_proba(X_test)

    # The first label in the chain has no context, so it must be identical.
    first = soft.order_[0]
    assert np.allclose(p_soft[:, first], p_hard[:, first], atol=1e-6)
    # Downstream labels see different context, so they should not all match.
    assert not np.allclose(p_soft, p_hard, atol=1e-6)


def test_label_powerset_marginals_are_consistent(synthetic_data):
    """LP marginals come from summing powerset classes, so they stay in [0, 1]."""
    X_train, Y_train, X_test, _ = synthetic_data
    lp = LabelPowerset(n_estimators=1, device="cpu", seed=0)
    lp.fit(X_train, Y_train)
    p = lp.predict_proba(X_test)
    assert p.shape == (len(X_test), 5)
    assert np.all(p >= -1e-6) and np.all(p <= 1.0 + 1e-6)
