"""
Classifier Chain (CC) strategy.

Factorises the joint label distribution via the chain rule of probability.
Each classifier is conditioned on the input features plus the ground-truth
labels of all preceding labels in the chain (during training):
    P(y | x) = Π_j P(y_{π(j)} | x, y_{π(1)}, …, y_{π(j-1)})

At inference, ground-truth labels are unavailable; each step propagates
the predicted probability (probabilistic=True) or the thresholded binary
prediction (probabilistic=False) as context to the next classifier.

Early errors in the chain can cascade downstream — ECC mitigates this.
"""
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.base import _get_tabpfn


class ClassifierChain(BaseEstimator, ClassifierMixin):
    """
    Sequential chain of TabPFN classifiers.

    Parameters
    ----------
    order : list of int, optional
        Label ordering for the chain. Defaults to [0, 1, …, L-1].
    probabilistic : bool
        If True, propagate the raw predicted probability as context at inference.
        If False, propagate a hard binary threshold (>= 0.5) instead.
    device : str
        Device for TabPFN.
    n_estimators : int
        Number of TabPFN ensemble members.
    seed : int
        Base random seed; each step gets seed + step_index.
    ignore_pretraining_limits : bool
        Passed to TabPFN.
    max_samples : int
        Subsample cap before fitting each step.
    """

    def __init__(
        self,
        order: Optional[List[int]] = None,
        probabilistic: bool = True,
        device: str = "auto",
        n_estimators: int = 16,
        seed: int = 42,
        ignore_pretraining_limits: bool = True,
        max_samples: int = 100_000,
    ):
        self.order = order
        self.probabilistic = probabilistic
        self.device = device
        self.n_estimators = n_estimators
        self.seed = seed
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.max_samples = max_samples

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ClassifierChain":
        """
        Fit each classifier in the chain on [X | ground-truth labels so far].
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        self.n_labels_ = Y.shape[1]
        self.order_ = self.order if self.order is not None else list(range(self.n_labels_))

        self.classifiers_ = []
        X_aug = X.copy()

        for step, label_idx in enumerate(self.order_):
            clf = _get_tabpfn(
                device=self.device,
                n_estimators=self.n_estimators,
                seed=self.seed + step,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
            )
            y_target = Y[:, label_idx]

            X_fit, y_fit = self._subsample(X_aug, y_target, self.seed + step)
            clf.fit(X_fit, y_fit)
            self.classifiers_.append(clf)

            # Append ground-truth context column for the next step
            if step < self.n_labels_ - 1:
                X_aug = np.column_stack([X_aug, y_target.astype(np.float32)])

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability matrix (N, L) using sequential chain inference.
        Each step's prediction is appended as context for the next step.
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        X_aug = X.copy()
        for step, label_idx in enumerate(self.order_):
            p = self.classifiers_[step].predict_proba(X_aug)[:, 1]
            probs[:, label_idx] = p

            if step < self.n_labels_ - 1:
                context = p if self.probabilistic else (p >= 0.5).astype(np.float32)
                X_aug = np.column_stack([X_aug, context])

        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def _subsample(self, X, y, seed):
        if len(X) > self.max_samples:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), self.max_samples, replace=False)
            return X[idx], y[idx]
        return X, y
