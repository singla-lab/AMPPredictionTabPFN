"""
Write the feature matrices out as CSV.

The matrices ship as Parquet (float32, zstd) because the equivalent CSVs are
150 MB each and two of them exceed GitHub's 100 MB per-file limit. Nothing in
this repository needs the CSVs -- ``src.data.loader`` resolves a request for
``final_test.csv`` to ``final_test.parquet`` automatically -- so this script is
only for interoperating with outside tooling that wants plain text.

The CSVs it writes are NOT byte-identical to the originals: features are stored
as float32, so a value that was printed with 16 significant digits comes back
with float32 precision. This is not a loss for the pipeline, because
``load_dataset`` casts features to float32 regardless, so the array the models
consume is bit-identical either way.

Usage:
    python scripts/materialize_features.py
    python scripts/materialize_features.py --dir data/final_v2
"""
import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import read_table

SPLITS = ("final_fold1", "final_fold2", "final_test")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/final",
                    help="directory holding the matrices (default: data/final)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite CSVs that already exist")
    args = ap.parse_args()

    d = (REPO_ROOT / args.dir) if not pathlib.Path(args.dir).is_absolute() else pathlib.Path(args.dir)
    if not d.is_dir():
        raise SystemExit(f"no such directory: {d}\nSee docs/DATA.md.")

    for split in SPLITS:
        dest = d / f"{split}.csv"
        if dest.exists() and not args.overwrite:
            print(f"  {split:12s} exists, skipping (use --overwrite)")
            continue
        df = read_table(d / f"{split}.csv")
        df.to_csv(dest, index=False)
        print(f"  {split:12s} {len(df):6,} rows x {df.shape[1]} cols "
              f"-> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
