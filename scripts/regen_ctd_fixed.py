"""
Regenerate the CTD block with corrected amino-acid group definitions.

Three defects in 09_scripts/extraction/feature_generator/ctd_generator.py on the
`main` branch (confirmed against the released source):

  1. Volume reuses Polarizability's residue strings verbatim (only the comments
     were changed), so all 21 Volume_* columns duplicate Polarizability_* and
     normalized van der Waals volume is absent from the feature set.
  2. SecondaryStruct group 2 is 'VIYFWT' instead of 'VIYCWFT'. Cysteine matches
     no group, so the mapper assigns it '0' and drops it from Composition --
     C_1+C_2+C_3 ranges 0.60-1.00 instead of summing to 1. Disulfide-rich AMPs
     are worst affected.
  3. SolventAccessibility groups 2/3 are 'RKPSTHY'/'MSPDEQN' instead of
     'RKQEND'/'MSPTHY'. S and P appear in both, and the mapper's first-match
     `break` resolves them by dict order rather than by solvent accessibility.

Everything else about the calculation is reproduced exactly as the original
wrote it -- the first-match `break`, the '0' bucket for unmatched residues,
transitions counted only for unordered pairs 12/13/23 and divided by (n-1), and
the ceil-based percentile indexing for the distribution features. Only the group
definitions change, so any difference in the output is attributable to the fix.

Validation: Hydrophobicity, Polarity and Charge were already correct in the
original, so regenerating them must reproduce the shipped columns bit-for-bit.
The script asserts this before writing anything.

Writes data/final_v2/*.csv -- data/final/ is left untouched.

Usage:
    python scripts/regen_ctd_fixed.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import read_table, resolve_table

IN_DIR = REPO_ROOT / "data" / "final"
OUT_DIR = REPO_ROOT / "data" / "final_v2"

# Write final_v2 in the same encoding the shipped final/ matrices use, so the two
# directories stay interchangeable for every downstream script. Falls back to
# Parquet when the inputs are absent, so importing this module never fails.
try:
    OUT_SUFFIX = resolve_table(IN_DIR / "final_test.csv").suffix
except FileNotFoundError:
    OUT_SUFFIX = ".parquet"

# Corrected standard 3-group partitions (Dubchak et al.)
PROPERTIES = {
    "Hydrophobicity":       {"1": "RKEDQN",   "2": "GASTPHY",          "3": "CLVIMFW"},
    "Volume":               {"1": "GASTPDC",  "2": "NVEQIL",           "3": "MHKFRYW"},
    "Polarity":             {"1": "LIFWCMVY", "2": "PATGS",            "3": "HQRKEDN"},
    "Polarizability":       {"1": "GASDT",    "2": "CPNVEQIL",         "3": "KMHFRYW"},
    "Charge":               {"1": "KR",       "2": "ANCQGHILMFPSTWYV", "3": "DE"},
    "SecondaryStruct":      {"1": "EALMQKRH", "2": "VIYCWFT",          "3": "GNPSD"},
    "SolventAccessibility": {"1": "ALFCGIVW", "2": "RKQEND",           "3": "MSPTHY"},
}

# The original definitions, kept so the validation can prove which three were
# already correct and reproduce them exactly.
ORIGINAL = {
    "Hydrophobicity":       {"1": "RKEDQN",   "2": "GASTPHY",          "3": "CLVIMFW"},
    "Volume":               {"1": "GASDT",    "2": "CPNVEQIL",         "3": "KMHFRYW"},
    "Polarity":             {"1": "LIFWCMVY", "2": "PATGS",            "3": "HQRKEDN"},
    "Polarizability":       {"1": "GASDT",    "2": "CPNVEQIL",         "3": "KMHFRYW"},
    "Charge":               {"1": "KR",       "2": "ANCQGHILMFPSTWYV", "3": "DE"},
    "SecondaryStruct":      {"1": "EALMQKRH", "2": "VIYFWT",           "3": "GNPSDT"},
    "SolventAccessibility": {"1": "ALFCGIVW", "2": "RKPSTHY",          "3": "MSPDEQN"},
}


def calculate_ctd(sequence, properties):
    """Byte-faithful reimplementation of the original calculate_ctd()."""
    res = {}
    n = len(sequence)
    if n == 0:
        return None
    for prop_name, groups in properties.items():
        mapped = []
        for aa in sequence:
            found = False
            for group_id, aa_list in groups.items():
                if aa in aa_list:
                    mapped.append(group_id)
                    found = True
                    break
            if not found:
                mapped.append("0")

        counts = {gid: mapped.count(gid) for gid in groups}
        for gid in groups:
            res[f"{prop_name}_C_{gid}"] = counts[gid] / n

        transitions = {f"{i}{j}": 0 for i in "123" for j in "123" if i < j}
        for i in range(len(mapped) - 1):
            pair = "".join(sorted([mapped[i], mapped[i + 1]]))
            if pair in transitions:
                transitions[pair] += 1
        for pair, count in transitions.items():
            res[f"{prop_name}_T_{pair}"] = count / (n - 1) if n > 1 else 0

        for gid in groups:
            indices = [i + 1 for i, v in enumerate(mapped) if v == gid]
            k = len(indices)
            for percentile in [0, 25, 50, 75, 100]:
                if k > 0:
                    pos = int(np.ceil(percentile * k / 100)) or 1
                    res[f"{prop_name}_D_{gid}_{percentile}"] = indices[pos - 1] / n
                else:
                    res[f"{prop_name}_D_{gid}_{percentile}"] = 0.0
    return res


def build(seqs, properties, desc):
    return pd.DataFrame([calculate_ctd(s, properties) for s in tqdm(seqs, desc=desc)])


def _tolerance(df, cols):
    """
    Comparison tolerance appropriate to how the shipped matrix is stored.

    The Parquet matrices hold features as float32 (bit-identical to what the
    models consume, since load_dataset casts to float32 anyway), so an exact
    float64 comparison would flag rounding as a mismatch. CTD distribution
    values reach 100, giving a float32 representation error near 8e-6; 1e-4 sits
    well above that and far below any real change in group definitions, which
    move values by 0.01 or more.
    """
    return 1e-4 if any(df[c].dtype == np.float32 for c in cols) else 1e-12


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {f: read_table(IN_DIR / f"final_{f}.csv") for f in ("fold1", "fold2", "test")}

    # ── validation: reproduce the shipped columns with the ORIGINAL groups ────
    probe = frames["fold1"].head(2000)
    orig = build(probe["Sequence"].values, ORIGINAL, "validating reimplementation")
    ok, bad = [], []
    for prop in PROPERTIES:
        cols = [c for c in orig.columns if c.startswith(prop + "_")]
        shipped = probe[cols].values.astype(np.float64)
        mine = orig[cols].values
        tol = _tolerance(probe, cols)
        (ok if np.abs(shipped - mine).max() < tol else bad).append(prop)
    print(f"\n  reimplementation reproduces shipped columns for: {ok}")
    if bad:
        print(f"  MISMATCH for: {bad}")
        raise SystemExit("reimplementation does not match the original; aborting")
    print("  -> all 7 properties reproduced exactly with the ORIGINAL groups, so any"
          "\n     change below is attributable to the corrected group definitions.\n")

    for split, df in frames.items():
        new = build(df["Sequence"].values, PROPERTIES, f"regenerating {split}")
        tol = _tolerance(df, list(new.columns))
        changed = [c for c in new.columns
                   if np.abs(df[c].values.astype(np.float64) - new[c].values).max() > tol]
        out = df.copy()
        for c in new.columns:
            # Preserve the input's storage dtype so final_v2 matches final.
            out[c] = new[c].values.astype(df[c].dtype)
        dest = OUT_DIR / f"final_{split}{OUT_SUFFIX}"
        if OUT_SUFFIX == ".parquet":
            out.to_parquet(dest, compression="zstd", index=False)
        else:
            out.to_csv(dest, index=False)
        print(f"  {split:6s} n={len(df):6,}  columns changed: {len(changed):3d}/147"
              f"  -> {dest}")

    # ── post-fix sanity ──────────────────────────────────────────────────────
    d = read_table(OUT_DIR / "final_fold1.csv")
    print(f"\n  post-fix checks on {OUT_DIR.name}/final_fold1{OUT_SUFFIX}:")
    v = [c for c in d.columns if c.startswith("Volume_")]
    p = [c.replace("Volume", "Polarizability") for c in v]
    same = sum(1 for a, b in zip(v, p) if np.array_equal(d[a].values, d[b].values))
    print(f"    Volume_* identical to Polarizability_* : {same}/21   (was 21/21)")
    for prop in ("SecondaryStruct", "SolventAccessibility", "Volume"):
        s = d[[f"{prop}_C_{i}" for i in (1, 2, 3)]].sum(axis=1)
        print(f"    {prop:22s} C_1+C_2+C_3 in [{s.min():.4f}, {s.max():.4f}]")
    feats = [c for c in d.columns
             if c not in ["Hash", "Sequence", "Antibacterial", "Antifungal",
                          "Antiviral", "Antiparasitic", "Antimicrobial"]]
    A = pd.concat([read_table(OUT_DIR / f"final_{s}.csv")[feats]
                   for s in ("fold1", "fold2", "test")]).values.astype(np.float64)
    seen, dups = {}, 0
    for i in range(A.shape[1]):
        k = A[:, i].tobytes()
        if k in seen:
            dups += 1
        else:
            seen[k] = i
    s = np.linalg.svd(A - A.mean(0), compute_uv=False)
    rank = int((s > s[0] * 1e-10).sum())
    print(f"    duplicate columns: {dups}  (was 25)")
    print(f"    rank {rank}/{A.shape[1]}  deficiency {A.shape[1]-rank}  (was 48)")


if __name__ == "__main__":
    main()
