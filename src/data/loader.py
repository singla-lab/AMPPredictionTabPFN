"""
Dataset loading and synthetic data generation for multilabel classification.

Feature matrices ship as Parquet (float32, zstd) because the equivalent CSVs are
150 MB each and two of them exceed GitHub's 100 MB per-file limit. ``load_dataset``
casts features to float32 anyway, so the Parquet copy yields a bit-identical
feature array while being ~4x smaller.

Paths are resolved by ``resolve_table``, so a request for ``final_test.csv``
transparently reads ``final_test.parquet`` when the CSV is absent. Existing call
sites therefore need no change, and ``scripts/materialize_features.py`` can write
the CSVs out if you want them on disk.
"""
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LABEL_COLS = ["Antibacterial", "Antifungal", "Antiviral", "Antiparasitic", "Antimicrobial"]

# Interchangeable on-disk encodings of the same table, in preference order.
_TABLE_SUFFIXES = (".parquet", ".csv")


def resolve_table(path) -> Path:
    """
    Resolve a feature-matrix path to whichever encoding is actually present.

    ``data/final/final_test.csv`` resolves to ``final_test.parquet`` when only the
    Parquet copy exists, and vice versa. An existing path is returned unchanged.

    Raises:
        FileNotFoundError: if no encoding of the table exists.
    """
    p = Path(path)
    if p.exists():
        return p
    for suffix in _TABLE_SUFFIXES:
        alt = p.with_suffix(suffix)
        if alt.exists():
            return alt
    tried = ", ".join(str(p.with_suffix(s).name) for s in _TABLE_SUFFIXES)
    raise FileNotFoundError(
        f"Dataset not found: {p}. Looked for {tried} in {p.parent}. "
        f"See docs/DATA.md for how to obtain the feature matrices."
    )


def read_table(path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Read a feature matrix as a DataFrame, in whichever encoding is on disk.

    Use this instead of ``pd.read_csv`` so that a Parquet-only checkout works.

    Args:
        columns: Optional column subset, equivalent to ``usecols`` for CSV.
    """
    p = resolve_table(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p, columns=list(columns) if columns else None)
    return pd.read_csv(p, usecols=list(columns) if columns else None)


def load_dataset(
    path: str,
    label_cols: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a feature matrix (Parquet or CSV) and return (X, Y, feature_names).

    Args:
        path: Path to the feature matrix. The .parquet/.csv suffix is
              interchangeable; whichever encoding exists on disk is read.
        label_cols: Target columns. Defaults to the 5 AMP activity labels in LABEL_COLS
                    if they are present in the file. Must be specified explicitly otherwise.
        feature_cols: Feature columns. If None, uses all numeric non-label columns.

    Returns:
        X (N, F) float32 feature matrix, Y (N, L) int32 label matrix, list of feature names.
    """
    df = read_table(path)

    if label_cols is None:
        label_cols = [c for c in LABEL_COLS if c in df.columns]
        if not label_cols:
            raise ValueError(
                f"No default label columns found in {path}. "
                f"Pass label_cols explicitly. Expected one or more of: {LABEL_COLS}"
            )

    if feature_cols is None:
        numeric = set(df.select_dtypes(include=[np.number]).columns)
        feature_cols = [c for c in df.columns if c in numeric and c not in label_cols]

    X = df[feature_cols].values.astype(np.float32)
    Y = df[label_cols].values.astype(np.int32)
    return X, Y, feature_cols


def make_synthetic_data(
    n_samples: int = 300,
    n_features: int = 20,
    n_labels: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a correlated multilabel dataset for local testing.
    70% train / 30% test split.

    Returns:
        X_train, Y_train, X_test, Y_test
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    W = rng.randn(n_features, n_labels)
    logits = X @ W + rng.randn(n_samples, n_labels) * 0.5
    Y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(np.int32)

    # Guarantee at least 2 classes per label so classifiers can fit
    for j in range(n_labels):
        if len(np.unique(Y[:, j])) < 2:
            half = n_samples // 2
            Y[:half, j] = 1
            Y[half:, j] = 0

    split = int(0.7 * n_samples)
    return X[:split], Y[:split], X[split:], Y[split:]
