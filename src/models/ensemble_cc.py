"""
Ensemble Classifier Chains (ECC) strategy.

Trains N chains each with a independently random label permutation π_k.
Final per-label probability is the average across all N chains:
    P(y_j | x) = (1/N) Σ_k P_{π_k}(y_j | x, ŷ_pred^(k))

Averaging over random orderings smooths out the cascade errors and
ordering bias inherent in a single Classifier Chain. As per the paper
settings we use N=8 chains with n_estimators=8 each.
"""
import gc
import os
import tempfile
import joblib
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.classifier_chain import ClassifierChain


class EnsembleClassifierChain(BaseEstimator, ClassifierMixin):
    """
    Average of N Classifier Chains with random label permutations.

    Parameters
    ----------
    n_chains : int
        Number of chains in the ensemble.
    probabilistic : bool
        Passed to each ClassifierChain (soft vs hard context at inference).
    device : str
        Device for TabPFN.
    n_estimators : int
        TabPFN ensemble members per chain. Default 8 (as per info.tex for ECC).
    seed : int
        Base seed; chain k uses seed + k.
    ignore_pretraining_limits : bool
        Passed to TabPFN.
    max_samples : int
        Subsample cap per chain step.
    """

    def __init__(
        self,
        n_chains: int = 8,
        probabilistic: bool = True,
        device: str = "auto",
        n_estimators: int = 8,
        seed: int = 42,
        ignore_pretraining_limits: bool = True,
        max_samples: int = 100_000,
    ):
        self.n_chains = n_chains
        self.probabilistic = probabilistic
        self.device = device
        self.n_estimators = n_estimators
        self.seed = seed
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.max_samples = max_samples

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "EnsembleClassifierChain":
        """Fit N chains, each with a unique random label ordering."""
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        self.n_labels_ = Y.shape[1]
        self._cache_dir = tempfile.TemporaryDirectory()
        self.chains_: List[str] = []  # Will store file paths instead of objects
        self.orders_: List[List[int]] = []

        for k in range(self.n_chains):
            rng = np.random.RandomState(self.seed + k)
            order = list(rng.permutation(self.n_labels_))
            chain = ClassifierChain(
                order=order,
                probabilistic=self.probabilistic,
                device=self.device,
                n_estimators=self.n_estimators,
                seed=self.seed + k * 100,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
                max_samples=self.max_samples,
            )
            chain.fit(X, Y)
            
            # Offload to disk to prevent CPU RAM OOM
            chain_path = os.path.join(self._cache_dir.name, f"chain_{k}.joblib")
            joblib.dump(chain, chain_path)
            self.chains_.append(chain_path)
            self.orders_.append(order)
            
            # Free RAM immediately
            del chain
            # Flush GPU allocator cache between chains to bound VRAM residual
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        return self

    def predict_proba(self, X: np.ndarray,
                      probabilistic: Optional[bool] = None) -> np.ndarray:
        """Average probabilities across all chains using an incremental
        accumulator to keep only ONE chain's output in RAM at a time.
        Returns (N, L) float32 array.

        Parameters
        ----------
        probabilistic : bool, optional
            Override the context representation each chain propagates at
            inference. None (default) keeps whatever the chains were built with,
            so existing behaviour is unchanged. Passing an explicit value lets a
            single fitted ensemble be queried both ways, which is what the
            soft-vs-hard context ablation needs — the chains fit on ground-truth
            binary context, so hard 0/1 propagation matches training while raw
            probabilities do not.
        """
        X = np.asarray(X, dtype=np.float32)
        acc: Optional[np.ndarray] = None
        for chain_path in self.chains_:
            chain = joblib.load(chain_path)
            if probabilistic is not None:
                chain.probabilistic = probabilistic
            probs = chain.predict_proba(X).astype(np.float32)
            del chain  # Free RAM
            gc.collect()

            if acc is None:
                acc = probs
            else:
                acc += probs   # accumulate in-place; only 2 arrays in RAM at once
                del probs
        return acc / self.n_chains  # type: ignore[operator]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
