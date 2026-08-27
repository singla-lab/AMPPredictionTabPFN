"""
Loader for the canonical prediction cache under ``data/preds_prob/``.

Each ``<strategy>/<strategy>_seed<seed>.npy`` is a pickled dict holding the
out-of-fold training probabilities and the test probabilities for one
(strategy, seed) run, plus a SHA-256 fingerprint over both arrays. See
``data/preds_prob/MANIFEST.json`` for which run produced each file.

Every post-hoc analysis reads through here rather than re-fitting TabPFN, so
the analyses stay consistent with one another and cost no GPU time.
"""
import hashlib
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.loader import LABEL_COLS, load_dataset

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PREDS_DIR = REPO_ROOT / "data" / "preds_prob"
DATA_DIR = REPO_ROOT / "data" / "final"

STRATEGIES = ["BR", "LP", "CC", "ECC", "PCC"]
SEEDS = [42, 1665, 8914]

# Antimicrobial is a deterministic OR of the other four, so dependence and
# headroom analyses run on the four real activities only.
ANALYSIS_LABELS = ["Antibacterial", "Antifungal", "Antiviral", "Antiparasitic"]
OR_PARENT = "Antimicrobial"


def _probs_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def load_run(
    strategy: str,
    seed: int,
    preds_dir: pathlib.Path = PREDS_DIR,
    verify: bool = True,
) -> dict:
    """
    Load one cached (strategy, seed) run.

    Args:
        verify: Recompute the SHA-256 over the arrays and raise if it disagrees
                with the fingerprint stored in the file.

    Returns
    -------
    Dict with 'strategy', 'seed', 'label_cols', 'train_oof_probs' (n_train, L)
    and 'test_probs' (n_test, L), both float32.
    """
    s = strategy.lower()
    path = pathlib.Path(preds_dir) / s / f"{s}_seed{seed}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached predictions at {path}. "
            "Run `python scripts/evaluate.py` first: it fits all five transforms for all three seeds and writes the cache. See docs/DATA.md section 3."
        )

    data = np.load(path, allow_pickle=True).item()
    oof = np.asarray(data["train_oof_probs"], dtype=np.float32)
    test = np.asarray(data["test_probs"], dtype=np.float32)

    if verify and "sha256" in data:
        actual = _probs_hash(oof, test)
        if actual != data["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: stored {data['sha256'][:12]}, "
                f"recomputed {actual[:12]}. The cache may be corrupt."
            )

    return {
        "strategy": data.get("strategy", strategy),
        "seed": int(data.get("seed", seed)),
        "label_cols": list(data.get("label_cols", LABEL_COLS)),
        "train_oof_probs": oof,
        "test_probs": test,
        "path": str(path),
    }


def load_test_probs(
    strategies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    preds_dir: pathlib.Path = PREDS_DIR,
) -> Dict[str, List[np.ndarray]]:
    """{strategy: [test probs, one (n_test, L) array per seed in `seeds` order]}."""
    strategies = strategies or STRATEGIES
    seeds = seeds or SEEDS
    return {
        s: [load_run(s, seed, preds_dir)["test_probs"] for seed in seeds]
        for s in strategies
    }


def load_oof_probs(
    strategies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    preds_dir: pathlib.Path = PREDS_DIR,
) -> Dict[str, List[np.ndarray]]:
    """{strategy: [OOF train probs, one (n_train, L) array per seed]}."""
    strategies = strategies or STRATEGIES
    seeds = seeds or SEEDS
    return {
        s: [load_run(s, seed, preds_dir)["train_oof_probs"] for seed in seeds]
        for s in strategies
    }


def load_labels(data_dir: pathlib.Path = DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ground-truth label matrices aligned with the cached probabilities.

    Training labels are fold1 then fold2 concatenated, matching how both
    ``generate_oof_probs`` and ``generate_2fold_oof_probs`` order their output.

    Returns
    -------
    (Y_train (n_train, L) int32, Y_test (n_test, L) int32)
    """
    data_dir = pathlib.Path(data_dir)
    _, Y1, feat = load_dataset(str(data_dir / "final_fold1.csv"))
    _, Y2, _ = load_dataset(str(data_dir / "final_fold2.csv"), feature_cols=feat)
    _, Y_te, _ = load_dataset(str(data_dir / "final_test.csv"), feature_cols=feat)
    return np.concatenate([Y1, Y2], axis=0), Y_te


def label_indices(labels: List[str], label_cols: List[str] = None) -> List[int]:
    """Column positions of `labels` within the full label ordering."""
    label_cols = label_cols or LABEL_COLS
    return [label_cols.index(c) for c in labels]
