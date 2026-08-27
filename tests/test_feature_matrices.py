"""
Integrity checks for the shipped feature matrices.

Skipped automatically when data/final/ is absent, so the suite still passes on a
checkout that has not fetched the data.
"""
import pathlib

import numpy as np
import pytest

from src.data import LABEL_COLS, load_dataset, read_table, resolve_table

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "final"

EXPECTED_ROWS = {"final_fold1": 32948, "final_fold2": 32922, "final_test": 16489}
N_FEATURES = 330

pytestmark = pytest.mark.skipif(
    not (DATA / "final_test.parquet").exists() and not (DATA / "final_test.csv").exists(),
    reason="feature matrices not present; see docs/DATA.md",
)


@pytest.mark.parametrize("split,rows", EXPECTED_ROWS.items())
def test_shape_and_schema(split, rows):
    X, Y, feat = load_dataset(str(DATA / f"{split}.csv"))
    assert X.shape == (rows, N_FEATURES)
    assert Y.shape == (rows, len(LABEL_COLS))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert set(np.unique(Y)) <= {0, 1}


def test_suffix_is_interchangeable():
    """A request for .csv resolves to whichever encoding is on disk."""
    assert resolve_table(DATA / "final_test.csv").stem == "final_test"
    assert resolve_table(DATA / "final_test.parquet").stem == "final_test"


def test_feature_columns_agree_across_splits():
    _, _, f1 = load_dataset(str(DATA / "final_fold1.csv"))
    _, _, f2 = load_dataset(str(DATA / "final_fold2.csv"))
    _, _, ft = load_dataset(str(DATA / "final_test.csv"))
    assert f1 == f2 == ft


def test_or_gate_holds_exactly():
    """Antimicrobial is the logical OR of the four activities, in every row."""
    for split in EXPECTED_ROWS:
        df = read_table(DATA / f"{split}.csv", columns=LABEL_COLS)
        children = df[[c for c in LABEL_COLS if c != "Antimicrobial"]].values
        assert np.array_equal(df["Antimicrobial"].values.astype(bool),
                              children.any(axis=1)), split


def test_no_sequence_overlap_between_splits():
    seqs = {s: set(read_table(DATA / f"{s}.csv", columns=["Sequence"]).Sequence)
            for s in EXPECTED_ROWS}
    assert not seqs["final_test"] & (seqs["final_fold1"] | seqs["final_fold2"])
    assert not seqs["final_fold1"] & seqs["final_fold2"]


def test_test_fold_label_counts_match_the_paper():
    df = read_table(DATA / "final_test.csv", columns=LABEL_COLS)
    assert df["Antibacterial"].sum() == 3187
    assert df["Antifungal"].sum() == 1316
    assert df["Antiviral"].sum() == 946
    assert df["Antiparasitic"].sum() == 77
    assert df["Antimicrobial"].sum() == 4235
