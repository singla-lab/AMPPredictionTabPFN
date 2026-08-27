"""
Binary Relevance (BR) strategy.

Trains L independent binary TabPFN classifiers, one per label.
Assumes all labels are conditionally independent given the input features:
    P(y | x) ≈ Π_j P(y_j | x)

This is the independence baseline — it ignores all label correlations.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.base import _get_tabpfn


class BinaryRelevance(BaseEstimator, ClassifierMixin):
    """
    Fit L independent TabPFN classifiers, one per label.

    Parameters
    ----------
    device : str
        Device for TabPFN (e.g. 'auto', 'cuda:0', 'cpu').
    n_estimators : int
        Number of TabPFN ensemble members.
    seed : int
        Base random seed; each label classifier gets seed + label_index.
    ignore_pretraining_limits : bool
        Passed to TabPFN to allow larger datasets.
    max_samples : int
        If the training set is larger than this, subsample before fitting.
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

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BinaryRelevance":
        """Fit one binary classifier per label column."""
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        self.n_labels_ = Y.shape[1]
        self.classifiers_ = []

        for j in range(self.n_labels_):
            clf = _get_tabpfn(
                device=self.device,
                n_estimators=self.n_estimators,
                seed=self.seed + j,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
            )
            X_fit, y_fit = self._subsample(X, Y[:, j], self.seed + j)
            clf.fit(X_fit, y_fit)
            self.classifiers_.append(clf)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix of shape (N, L)."""
        X = np.asarray(X, dtype=np.float32)
        return np.column_stack(
            [clf.predict_proba(X)[:, 1] for clf in self.classifiers_]
        )

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def _subsample(self, X, y, seed):
        if len(X) > self.max_samples:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), self.max_samples, replace=False)
            return X[idx], y[idx]
        return X, y
