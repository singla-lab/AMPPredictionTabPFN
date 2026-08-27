"""
Label Powerset (LP) strategy.

Maps each unique observed label combination to a distinct multiclass ID,
fits a single multiclass TabPFN, then recovers per-label marginals by
summing the powerset class probabilities where label j is active:
    P(y_j = 1 | x) = Σ_{c ∈ C : c_j = '1'} P(c | x)

LP captures all label correlations but is bounded by data scarcity for
rare combinations. With L=5 AMP activities only 16 of the 32 possible
label vectors appear (due to the deterministic Antimicrobial label), so
the multiclass problem is well-conditioned.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.base import _get_tabpfn


class LabelPowerset(BaseEstimator, ClassifierMixin):
    """
    Single multiclass TabPFN over observed label combinations + marginalization.

    Parameters
    ----------
    device : str
        Device for TabPFN.
    n_estimators : int
        Number of TabPFN ensemble members.
    seed : int
        Random seed.
    ignore_pretraining_limits : bool
        Passed to TabPFN.
    max_samples : int
        Subsample cap before fitting.
    """

    def __init__(
        self,
        device: str = "auto",
        n_estimators: int = 16,
        seed: int = 42,
        ignore_pretraining_limits: bool = True,
        max_samples: int = 100_000,
    ):
        self.device = device
        self.n_estimators = n_estimators
        self.seed = seed
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.max_samples = max_samples

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LabelPowerset":
        """Encode label combinations as multiclass IDs and fit a single classifier."""
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        self.n_labels_ = Y.shape[1]

        # Encode each row as a bitstring e.g. "10101"
        bitstrings = ["".join(row.astype(str)) for row in Y]
        unique_classes = sorted(set(bitstrings))
        self.class_to_id_ = {c: i for i, c in enumerate(unique_classes)}
        self.id_to_bits_ = {i: c for i, c in enumerate(unique_classes)}

        y_mc = np.array([self.class_to_id_[b] for b in bitstrings], dtype=np.int32)

        self.classifier_ = _get_tabpfn(
            device=self.device,
            n_estimators=self.n_estimators,
            seed=self.seed,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )

        if len(X) > self.max_samples:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(X), self.max_samples, replace=False)
            self.classifier_.fit(X[idx], y_mc[idx])
        else:
            self.classifier_.fit(X, y_mc)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict per-label marginal probabilities by marginalizing the powerset distribution:
            P(y_j = 1 | x) = Σ_{c : c_j = '1'} P(c | x)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]

        mc_probs = self.classifier_.predict_proba(X)
        fitted_classes = getattr(self.classifier_, "classes_", np.arange(mc_probs.shape[1]))

        P = np.zeros((n_samples, self.n_labels_), dtype=np.float32)
        for k, class_id in enumerate(fitted_classes):
            bits = self.id_to_bits_.get(class_id, "")
            for j in range(self.n_labels_):
                if j < len(bits) and bits[j] == "1":
                    P[:, j] += mc_probs[:, k]

        return P

    def predict_joint(self, X: np.ndarray) -> np.ndarray:
        """
        Full joint posterior over all 2^L label combinations, shape (N, 2^L).

        LP predicts a distribution over the label combinations it observed in
        training. This method scatters that distribution onto the complete
        2^L grid, in the canonical ordering used by
        src/utils/prioritization.py and src/models/probabilistic_cc.py
        (``itertools.product([0, 1], repeat=L)``, label 0 varying slowest).
        Combinations never observed in training receive probability 0, so rows
        already sum to 1 without renormalisation.
        """
        import itertools

        X = np.asarray(X, dtype=np.float32)
        combos = list(itertools.product([0, 1], repeat=self.n_labels_))
        combo_to_idx = {c: i for i, c in enumerate(combos)}

        mc_probs = self.classifier_.predict_proba(X)
        fitted = getattr(self.classifier_, "classes_", np.arange(mc_probs.shape[1]))

        joint = np.zeros((X.shape[0], len(combos)), dtype=np.float64)
        for k, class_id in enumerate(fitted):
            bits = self.id_to_bits_.get(class_id)
            if bits is None:
                continue
            joint[:, combo_to_idx[tuple(int(b) for b in bits)]] += mc_probs[:, k]
        return joint

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
