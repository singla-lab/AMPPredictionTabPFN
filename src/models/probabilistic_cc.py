"""
Probabilistic Classifier Chain (PCC) strategy.

Resolves the greedy inference limitation of CC by performing exact inference.
Exhaustively evaluates the chain rule over all 2^L possible label vectors,
then marginalises to Bayes-optimal per-label probabilities:
    P(y_j = 1 | x) = Σ_{y ∈ {0,1}^L : y_j=1} [ Π_i P(y_i | x, y_1, …, y_{i-1}) ]

Training is identical to a standard ClassifierChain (ground-truth context).
The computational cost is 2^L forward passes at inference — feasible for L ≤ 5.
"""
from typing import List, Optional

import itertools
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.classifier_chain import ClassifierChain


class ProbabilisticClassifierChain(BaseEstimator, ClassifierMixin):
    """
    Exact joint inference over 2^L label combinations.

    Parameters
    ----------
    order : list of int, optional
        Chain ordering. Defaults to [0, 1, …, L-1].
    device : str
        Device for TabPFN.
    n_estimators : int
        Number of TabPFN ensemble members.
    seed : int
        Random seed.
    ignore_pretraining_limits : bool
        Passed to TabPFN.
    max_samples : int
        Subsample cap before fitting each chain step.
    """

    def __init__(
        self,
        order: Optional[List[int]] = None,
        device: str = "auto",
        n_estimators: int = 16,
        seed: int = 42,
        ignore_pretraining_limits: bool = True,
        max_samples: int = 100_000,
    ):
        self.order = order
        self.device = device
        self.n_estimators = n_estimators
        self.seed = seed
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.max_samples = max_samples

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ProbabilisticClassifierChain":
        """Fit a standard chain with ground-truth context (same as ClassifierChain)."""
        self._chain = ClassifierChain(
            order=self.order,
            probabilistic=True,  # not used at inference for PCC
            device=self.device,
            n_estimators=self.n_estimators,
            seed=self.seed,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            max_samples=self.max_samples,
        )
        self._chain.fit(X, Y)
        self.n_labels_ = self._chain.n_labels_
        self.order_ = self._chain.order_
        return self

    def predict_joint(self, X: np.ndarray) -> np.ndarray:
        """
        Full joint posterior over all 2^L label combinations, shape (N, 2^L).

        This is the quantity predict_proba marginalises; it is returned directly
        here so that downstream analyses can condition on observed labels
        (see src/utils/prioritization.py) without a second inference pass.
        Combination ordering is ``itertools.product([0, 1], repeat=L)``.
        """
        return self._joint_and_marginals(X)[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Enumerate all 2^L binary combinations, compute the exact joint probability
        for each via the chain rule, normalise, then marginalise per label.

        Optimised via Dynamic Programming (caching independent prefixes) and
        multi-threading to concurrently dispatch independent TabPFN predictions
        to the GPU.
        """
        return self._joint_and_marginals(X)[1]

    def _joint_and_marginals(self, X: np.ndarray):
        """Shared implementation returning (joint (N, 2^L), marginals (N, L))."""
        import concurrent.futures

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        classifiers = self._chain.classifiers_

        # memo[step][prefix] = probability array of shape (n_samples,) for P(y_{step} = 1 | prefix)
        # prefix is a tuple of bits (0 or 1) of length `step`.
        memo = {}

        # Max threads = 16 (since step 4 has 16 prefixes)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for step in range(self.n_labels_):
                memo[step] = {}
                # All possible prefixes of length `step`
                prefixes = list(itertools.product([0, 1], repeat=step))
                
                def eval_prefix(prefix_tuple):
                    # Build X_aug for this specific prefix
                    if step == 0:
                        X_aug = X
                    else:
                        aug_cols = np.full(
                            (n_samples, step), 
                            np.array(prefix_tuple, dtype=np.float32)
                        )
                        X_aug = np.column_stack([X, aug_cols])
                    
                    # Return P(y_{current} = 1 | prefix)
                    return classifiers[step].predict_proba(X_aug)[:, 1]

                # Submit all independent prefixes to the GPU concurrently
                future_to_prefix = {
                    executor.submit(eval_prefix, p): p 
                    for p in prefixes
                }
                
                # Gather results as they complete
                for future in concurrent.futures.as_completed(future_to_prefix):
                    prefix = future_to_prefix[future]
                    memo[step][prefix] = future.result()

        # Compute exact joint probabilities using the cached conditionals
        all_combos = list(itertools.product([0, 1], repeat=self.n_labels_))
        joint = np.zeros((n_samples, len(all_combos)), dtype=np.float64)

        for c_idx, combo in enumerate(all_combos):
            p_joint = np.ones(n_samples, dtype=np.float64)
            for step, label_idx in enumerate(self.order_):
                # The prefix that lead to this step is the historical values in chain order
                prefix = tuple(combo[self.order_[i]] for i in range(step))
                bit = combo[label_idx]
                
                p1 = memo[step][prefix]
                p_joint *= p1 if bit == 1 else (1.0 - p1)
                
            joint[:, c_idx] = p_joint

        # Normalise for numerical stability
        joint /= joint.sum(axis=1, keepdims=True) + 1e-12

        # Marginalise: P(y_j = 1 | x) = sum of joint probs where combo[j] == 1
        marginals = np.zeros((n_samples, self.n_labels_), dtype=np.float32)
        for c_idx, combo in enumerate(all_combos):
            for j in range(self.n_labels_):
                if combo[j] == 1:
                    marginals[:, j] += joint[:, c_idx]

        return joint, marginals

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
