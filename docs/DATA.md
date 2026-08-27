# Data

**The feature matrices are tracked and ship with the repository.** Everything
else here is bulk or regenerable and is not tracked. This file says where each
remaining input comes from and what shape it must have.

```
data/
├── final/                          # TRACKED - 330-descriptor matrices, the paper's inputs
│   ├── final_fold1.parquet         #  32,948 peptides,  39 MB
│   ├── final_fold2.parquet         #  32,922 peptides,  39 MB
│   ├── final_test.parquet          #  16,489 peptides,  20 MB, held out
│   └── CHECKSUMS.json              #  SHA-256, row/column counts, label counts
├── final_v2/                       # same, with the corrected CTD block (regen_ctd_fixed.py)
├── preds_prob/                     # canonical probability cache, written by scripts/evaluate.py
│   ├── br/br_seed{42,1665,8914}.npy
│   ├── lp/…  cc/…  ecc/…  pcc/…
│   └── MANIFEST.json
└── escape_repro/                   # only needed for §3.2 and the structural ablation
    ├── ckpt/Best_model_Fold1.pth
    ├── ckpt/Best_model_Fold2.pth
    └── maps_test/<hash>.npy        # 5,495 released structural maps
```

---

## 1. The ESCAPE benchmark

82,359 peptides, five labels, with a fixed fold split. Used **unmodified**. You
only need this section if you want the raw sequences and the original release;
the derived feature matrices in Section 2 already ship with this repository.

- Data: <https://doi.org/10.7910/DVN/C69MCD> (Harvard Dataverse)
- Code and checkpoints: <https://github.com/BCV-Uniandes/ESCAPE>

Split, exactly as released:

| Split | Peptides | Role |
|---|---|---|
| Fold 1 | 32,948 | in-context samples |
| Fold 2 | 32,922 | in-context samples |
| Test | 16,489 | held out, testing only |

Labels are `Antibacterial`, `Antifungal`, `Antiviral`, `Antiparasitic`, and the
deterministic parent `Antimicrobial` = OR of the other four. That relation holds
exactly in every row of every split; `src/utils/orgate.py` verifies it.

Test-fold prevalence: antibacterial 19.33%, antifungal 7.98%, antiviral 5.74%,
antiparasitic 0.47% (77 positives), antimicrobial 25.68%.

---

## 2. Feature matrices — `data/final/*.parquet`

**These ship with the repository — you do not need to build or download them.**

### Why Parquet, and why float32

The equivalent CSVs are 156 MB, 116 MB and 78 MB; two of them exceed GitHub's
hard 100 MB per-file limit, so plain CSV cannot be tracked without Git LFS.
Parquet with float32 features and zstd compression brings the three files to
39 / 39 / 20 MB, all comfortably under the limit and with no LFS quota to
exhaust on clone.

float32 is **lossless for this pipeline**, not a compromise: `load_dataset`
executes `X = df[feature_cols].values.astype(np.float32)`, so the feature array
the models consume is bit-identical whether it came from the float64 CSV or the
float32 Parquet. This was verified column-by-column over the whole test fold
before the matrices were committed. The one place it matters is
`scripts/regen_ctd_fixed.py`, which compares recomputed CTD values against the
shipped ones; it selects a float32-appropriate tolerance automatically.

### Using them

Nothing needs changing: paths are resolved by suffix, so a request for
`final_test.csv` reads `final_test.parquet` when only the Parquet exists.

```python
from src.data import load_dataset, read_table

X, Y, feature_names = load_dataset("data/final/final_test.csv")   # (16489, 330), (16489, 5)
df = read_table("data/final/final_test.csv", columns=["Hash", "Sequence"])
```

To write plain CSVs for outside tooling:

```bash
python scripts/materialize_features.py            # data/final/*.csv
python scripts/materialize_features.py --dir data/final_v2
```

`data/final/CHECKSUMS.json` records the SHA-256, row and column counts, and
per-label positive counts of each file. `tests/test_feature_matrices.py` checks
the shapes, the OR-gate relation, the absence of cross-split sequence overlap,
and that the test-fold label counts match the paper.

### Schema

Each file is one row per peptide and **337 columns**:

| Columns | Contents |
|---|---|
| `Hash` | `sha256(Sequence)[:16]`, the peptide key used to join everything |
| 330 feature columns | the descriptor matrix, in the order below |
| `Sequence` | the amino-acid sequence |
| 5 label columns | `Antibacterial`, `Antifungal`, `Antiviral`, `Antiparasitic`, `Antimicrobial` |

`src/data/loader.py` takes every numeric non-label column as a feature, so
column order does not matter to the models — but it is identical across the three
files, and `tests/test_feature_matrices.py` asserts it.

### The 330 descriptors

| Family | Columns | Naming | How it was produced |
|---|---:|---|---|
| **PhysChem** | 10 | `Length`, `Charge`, `ChargeDensity`, `pI`, `InstabilityInd`, `Aromaticity`, `AliphaticInd`, `BomanInd`, `HydrophRatio`, `HydrophobicMoment` | the `peptides` library, with sequence-level scalars |
| **AAC** | 20 | `AAC_{A..Y}` | amino-acid composition, `propy3` |
| **CTD** | 147 | `{Property}_C_{1..3}`, `{Property}_T_{12,13,23}`, `{Property}_D_{1..3}_{0,25,50,75,100}` for 7 properties: Hydrophobicity, Volume, Polarity, Polarizability, Charge, SecondaryStruct, SolventAccessibility | `scripts/regen_ctd_fixed.py` |
| **ESM** | 128 | `ESM_PCA_{0..127}` | ESM-2 embeddings, mean-pooled over residues, then PCA |
| **AAIndex** | 15 | `AAIndex_PCA_{0..14}` | mean AAindex1 residue-property values per sequence, then PCA |
| **Motif** | 5 | `motif_R_X_R`, `motif_RW_repeat`, `motif_RGD`, `motif_C_X3_C`, `motif_RKKR` | regex counts |
| **Density** | 4 | `dens_Arg`, `dens_Trp`, `dens_Cys`, `dens_Pro` | residue count / length |
| **Cleavage** | 1 | `cleavage_basic_pair_count` | count of adjacent basic residue pairs |

CTD reads as three complementary views of one physicochemical property along the
sequence: **composition** (`_C_`) is the fraction of residues in each property
class, **transition** (`_T_`) the frequency of switches between classes at
adjacent positions, and **distribution** (`_D_`) the sequence positions at which
the first, 25%, 50%, 75% and 100% of a class are reached. So
`Hydrophobicity_D_3_0` is a *spatial* descriptor, not a compositional one.

### Regenerating the CTD block

```bash
python scripts/regen_ctd_fixed.py     # data/final/ -> data/final_v2/
```

Output is written in whatever encoding `data/final/` uses, so the two directories
stay interchangeable.

This corrects three amino-acid group definitions in the original CTD generator
(Volume duplicating Polarizability; a missing cysteine in SecondaryStruct group 2;
overlapping SolventAccessibility groups 2/3). Every other detail of the
calculation is reproduced exactly, so any difference in output is attributable to
the fix. 38 of the 330 columns change.

**The other seven families are not regenerated by code in this repository.** They
were produced by a separate extraction pipeline built on `propy3` (AAC), the
`peptides` library (PhysChem), mean-pooled ESM-2 embeddings reduced by PCA, and
the AAindex1 database from GenomeNet (<https://www.genome.jp/ftp/db/community/aaindex/aaindex1>),
also PCA-reduced. The table above gives the definitions needed to reconstruct
them; open an issue if you need the raw matrices.

---

## 3. Prediction cache — `data/preds_prob/`

Written by `scripts/evaluate.py`. One `.npy` per (transform, seed) holding a
pickled dict:

| Key | Shape | Contents |
|---|---|---|
| `train_oof_probs` | (65870, 5) | 5-fold out-of-fold probabilities over the in-context set, used to freeze honest thresholds |
| `test_probs` | (16489, 5) | test-set probabilities |
| `sha256` | — | fingerprint over both arrays |

Read it through `src/data/preds_cache.py`, which verifies the fingerprint on
load. Most analyses in `docs/PAPER_MAP.md` are cache-only: once `evaluate.py`
has run once, they cost no GPU time and stay consistent with one another.

---

## 4. ESCAPE checkpoints and structural maps — `data/escape_repro/`

Needed only for §3.2 (the paired comparison and fold-matched protocol) and the
structural-branch ablation in §3.4.

- `ckpt/Best_model_Fold1.pth`, `ckpt/Best_model_Fold2.pth` — the released
  checkpoints, used **without any retraining**.
- `maps_test/<hash>.npy` — the released structural maps.

The Drive release ships two map folders that disagree on modality:
`distance_maps_Test` holds int64 0/1 **contact** maps for 5,495 hashes, all in
the test split; `Distance_Maps` holds float32 **distance** maps in ångström
spread across folds. `scripts/escape_inference.py --mode modality_check` decides
which the checkpoints were actually trained on by measurement rather than by
folder name — run it first.

Coverage is 5,495 of 16,489 test peptides (33.3%), so every paired analysis
against ESCAPE is restricted to that subset, and ESCAPE's published whole-test
number cannot be reproduced from the release.

---

## 5. MMseqs2

`scripts/homology_analysis.py` needs the `mmseqs` binary on `PATH`:

```bash
conda install -c bioconda "mmseqs2=18.8cc5c"
```

Protocol: every test sequence is searched against the full in-context set with
`easy-search -s 7.5 -e 10000 --max-seqs 4000 --alignment-mode 3`; each test
peptide takes the maximum sequence identity among hits with query coverage
≥ 0.5, and peptides with no qualifying hit are assigned zero. There are no
exactly duplicated sequences within or across splits.

```bash
python scripts/homology_analysis.py --keep_tsv    # retains the ~167 MB hit table
```

Keep the table if you also want the per-label identity joins; it is regenerable
at any time and is deliberately not tracked.
