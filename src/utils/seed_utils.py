"""
seed_utils.py — global RNG seeding for full experiment reproducibility.

Call set_global_seeds(seed) at the top of any script to ensure that
Python's random, NumPy, and PyTorch all use the same seed, and that
CUDA operations are deterministic where possible.
"""
import os
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """
    Seed Python random, NumPy, and (if available) PyTorch CPU + CUDA RNGs.

    Also sets PYTHONHASHSEED via os.environ so that dict iteration order and
    hash-based operations are reproducible across processes (takes effect on
    the *next* Python subprocess, not the current one — call this early).

    Parameters
    ----------
    seed : int
        Master seed value. Individual per-label / per-fold seeds in the
        training loop should be derived as  seed + offset  to preserve
        independence between parallel components while remaining
        deterministically reproducible from this single master seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic algorithms where supported.
            # Note: some CUDA ops have no deterministic implementation;
            # they will raise RuntimeError only if use_deterministic_algorithms
            # is True AND warn_only is False.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng_states() -> dict:
    """
    Snapshot the current state of all RNGs (Python, NumPy, PyTorch).

    Returns a JSON-serialisable dict that can be saved to a run log.
    Restoring these states exactly would reproduce any subsequent random
    draws in the same process.
    """
    state: dict = {
        "python_random_state": list(random.getstate()[1]),   # tuple → list for JSON
        "numpy_random_state": {
            "name": "MT19937",
            "keys": np.random.get_state()[1].tolist(),
            "pos":  int(np.random.get_state()[2]),
        },
    }

    try:
        import torch
        state["torch_rng_state"] = torch.get_rng_state().tolist()
        if torch.cuda.is_available():
            # One entry per GPU
            state["torch_cuda_rng_states"] = [
                torch.cuda.get_rng_state(i).tolist()
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    return state
