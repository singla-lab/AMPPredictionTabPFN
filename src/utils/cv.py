"""
Cross-validation utility for generating out-of-fold (OOF) predictions.

Used for honest threshold selection — thresholds are chosen on OOF predictions
(no leakage), then frozen before the test set is evaluated.
"""
import gc
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold


def _free_gpu():
    """Release Python-managed objects and flush the CUDA allocator cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def generate_oof_probs(
    model_factory: Callable,
    X: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates out-of-fold probability predictions via k-fold cross-validation.

    Each fold's model is explicitly deleted and GPU memory flushed after
    prediction to avoid accumulating VRAM across folds.

    Args:
        model_factory: Callable with no arguments returning a fresh unfitted
                       model instance with the same hyperparameters each call.
        X: Feature matrix (N, F), float32.
        Y: Label matrix (N, L), int32.
        n_splits: Number of CV folds (default 5).
        seed: Random seed for fold splitting.

    Returns:
        oof_probs: Array of shape (N, L) with OOF predicted probabilities.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int32)
    n_samples, n_labels = Y.shape
    oof_probs = np.zeros((n_samples, n_labels), dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"    OOF fold {fold_idx + 1}/{n_splits} ...", flush=True)
        model = model_factory()
        model.fit(X[train_idx], Y[train_idx])
        oof_probs[val_idx] = model.predict_proba(X[val_idx])

        # Free fold model from CPU + GPU memory before next fold
        del model
        _free_gpu()

    return oof_probs
