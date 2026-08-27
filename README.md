# AMPPredictionTabPFN

Code for **"Coarse composition suffices: tabular in-context learning for
multi-activity antimicrobial peptide profiling."**

> 📄 **Preprint:** _bioRxiv link to be added._
> 📊 **Benchmark:** ESCAPE — [Harvard Dataverse](https://doi.org/10.7910/DVN/C69MCD) · [BCV-Uniandes/ESCAPE](https://github.com/BCV-Uniandes/ESCAPE)

---

## What this is

A sequence-only pipeline for predicting the **multi-activity profile** of an
antimicrobial peptide — antibacterial, antifungal, antiviral, antiparasitic, and
the deterministic antimicrobial parent — on the ESCAPE benchmark (82,359
peptides, five labels).

330 interpretable sequence descriptors are paired with **TabPFN**, a tabular
foundation model that takes the labelled reference peptides as *in-context
samples* and returns calibrated posteriors in a single forward pass. There is no
gradient training on peptide data and no hyperparameter search at any stage.

Five standard multi-label transforms are compared on an identical feature
matrix, split, and backbone, differing only in how the five-label target is
factorized:

| Transform | Idea | Fits | Inference passes |
|---|---|---|---|
| **BR** — Binary relevance | one classifier per label; conditional-independence floor | $L$ | $L$ |
| **LP** — Label powerset | the whole activity profile is one class | 1 | 1 |
| **CC** — Classifier chain | labels predicted in order, one committed path | $L$ | $L$ |
| **ECC** — Ensemble classifier chains | 8 chains under sampled label orders, averaged | $8L$ | $8L$ |
| **PCC** — Probabilistic classifier chain | full joint over all $2^L$ profiles | $L$ | $2^L$ |

**Headline results.** Label powerset reaches five-label macro average precision
of **77.79%** against **72.12%** for the previously best-performing method, at
the *lowest* cost of the five transforms. The probabilistic classifier chain is
the first method to match or exceed the best published average precision on all
five labels simultaneously. The margin survives the prior state of the art's
single-fold protocol, and is largest for remote homologues (+11.2 points below
30% sequence identity).

---

## Quick start

```bash
git clone https://github.com/singla-lab/AMPPredictionTabPFN.git
cd AMPPredictionTabPFN

conda env create -f environment.yml
conda activate amptabpfn
pip install -e .

pytest -q                      # unit tests for the transforms, metrics, OR-gate
```

pip-only alternative:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

**The feature matrices ship with the repository.** `data/final/` holds the
330-descriptor matrices for all three splits as Parquet, ready to use — no
download, no feature extraction, no preprocessing:

| File | Peptides | Size |
|---|---:|---:|
| `data/final/final_fold1.parquet` | 32,948 | 39 MB |
| `data/final/final_fold2.parquet` | 32,922 | 39 MB |
| `data/final/final_test.parquet` | 16,489 | 20 MB |

Each is one row per peptide with `Hash`, the 330 descriptors, `Sequence`, and the
five labels. So the smallest end-to-end run is just:

```bash
python scripts/train.py --strategy lp --seed 42     # one transform, one seed
```

Parquet rather than CSV because the equivalent CSVs are 150 MB each and two of
them exceed GitHub's 100 MB per-file limit. Features are stored as float32, which
is **lossless for this pipeline** — `load_dataset` casts to float32 regardless,
so the array the models consume is bit-identical to the original CSVs. Paths are
resolved by suffix, so `load_dataset("data/final/final_test.csv")` reads the
Parquet file transparently and no existing code path changes. If you want CSVs on
disk for outside tooling:

```bash
python scripts/materialize_features.py
```

Only the ESCAPE checkpoints and structural maps (needed for §3.2 and the
structural ablation) must be fetched separately — see
**[docs/DATA.md](docs/DATA.md)**. Nothing under `results/` is tracked;
`docs/PAPER_MAP.md` says which script writes which output.

---

## Reproducing the paper

Every command below is run from the repository root. Scripts take `--device`
where they need a GPU; all of them write to `results/` and none overwrite an
existing file. `docs/PAPER_MAP.md` maps each table, figure, and quoted number in
the manuscript to the exact script and output artefact.

### Stage 0 — features and splits

Nothing to do: `data/final/` ships ready to use. Optionally regenerate the CTD
block (147 of the 330 columns) with corrected amino-acid group definitions,
which writes `data/final_v2/`:

```bash
python scripts/regen_ctd_fixed.py
```

### Stage 1 — the five-transform benchmark

*§3.1 "Improved average precision across all activity labels" — the frontier and
cost tables, both per-label figures, and the per-seed supplementary table.*

```bash
python scripts/evaluate.py --device cuda:0            # 5 transforms x 3 seeds, GPU, hours
python scripts/collate_benchmark.py                   # per-seed table from the caches
python scripts/paired_ci_analysis.py                  # seed-averaged bootstrap CIs
```

`evaluate.py` writes the canonical probability cache under `data/preds_prob/`
(5 transforms × 3 seeds, SHA-256 fingerprinted). **Every analysis below reads
that cache instead of re-fitting**, so once Stage 1 has run they are cheap and
mostly CPU-only.

### Stage 2 — ESCAPE baseline and the fold-matched protocol

*§3.2 "Separating data-size effects from model gains" — the paired-comparison table.*

```bash
python scripts/escape_inference.py --mode modality_check --device cuda:0
python scripts/escape_inference.py --mode test_subset    --device cuda:0
python scripts/escape_paired_compare.py                       # paired bootstrap, 5,495 peptides
python scripts/fold_matched_protocol.py --device cuda:0       # one fold at a time as context
```

### Stage 3 — homology stratification

*§3.3 "Remote-homologue peptides show strongest gains" — the identity-band table.*

Needs MMseqs2 on `PATH` (`conda install -c bioconda "mmseqs2=18.8cc5c"`).

```bash
python scripts/homology_analysis.py --sensitivity 7.5 --min_qcov 0.5
```

### Stage 4 — is sequence enough?

*§3.4 "Sequence-only information is sufficient" — the feature-group attribution
table and the structural-branch ablation.*

```bash
python scripts/feature_group_attribution.py --device cuda:0   # group-only / LOO / permutation
python scripts/escape_structure_ablation.py --device cuda:0   # zero / noise / swapped maps
```

### Stage 5 — label dependence

*§3.5 "Dependence helps mainly for scarce activities" — the dependence figure and
the OR-gate coherence table.*

```bash
python scripts/oracle_dependence_ceiling.py --device cuda:0   # oracle headroom, per label
python scripts/joint_likelihood_dependence.py --device cuda:0 # joint vs product log-likelihood
python scripts/cc_context_ablation.py --device cuda:0         # CC soft vs hard chain context
python scripts/cc_hard_followups.py                           # CC-hard oracle recovery
python scripts/orgate_analysis.py                             # OR-gate violation rates
```

### Stage 6 — next-assay prioritization

*§3.6 "Ranking activities from partial positive evidence" — the assay
prioritization table.*

```bash
python scripts/pu_next_assay.py                               # Hit@1 / MRR / PU-AP, m = 0,1,2
```

### Supplementary — context-size scaling

*Supplementary "Scaling with the number of in-context samples".*

```bash
python scripts/scaling_analysis.py --strategy all             # n = 250 ... 65,870
```

---

## Repository layout

| Path | Contents |
|---|---|
| `src/models/` | the five multi-label transforms (`BinaryRelevance`, `LabelPowerset`, `ClassifierChain`, `EnsembleClassifierChain`, `ProbabilisticClassifierChain`) over a shared TabPFN backbone |
| `src/data/` | dataset loading (`loader.py`) and the SHA-256-verified prediction cache reader (`preds_cache.py`) |
| `src/utils/` | average precision and threshold metrics, bootstrap CIs, out-of-fold CV, OR-gate coherence, prioritization ranking, seeding |
| `scripts/` | one script per paper analysis; see `docs/PAPER_MAP.md` |
| `external/escape/` | vendored ESCAPE model code, for loading the released checkpoints — see `external/escape/NOTICE.md` |
| `tests/` | unit tests for the transforms, metrics and OR-gate utilities |
| `data/final/` | the 330-descriptor feature matrices for all three splits, ready to use |
| `docs/DATA.md` | the descriptor schema, and how to obtain the ESCAPE checkpoints and structural maps |
| `docs/PAPER_MAP.md` | every manuscript table/figure → script → output artefact |

---

## Protocol notes

- **Split.** The predefined ESCAPE split is used unmodified: folds 1–2 (65,870
  peptides) supply the in-context samples, the held-out fold (16,489) is used
  only for testing.
- **Seeds.** `{42, 1665, 8914}` throughout, matching the published baseline's
  seed protocol. Point estimates are means over the three seeds.
- **Backbone.** TabPFN v3 (`tabpfn` 8.2.0), `n_estimators = 16` (8 for ECC, so
  that eight chains stay affordable). No hyperparameter was tuned on ESCAPE at
  any stage.
- **"Fit" vs "inference".** A *fit* performs no weight updates — it loads the
  labelled reference peptides into the model context. *Inference* is the single
  forward pass that reads a test peptide together with that context. Reported
  fit time is context ingestion, not optimisation.
- **Thresholds.** Honest per-label thresholds are chosen on 5-fold out-of-fold
  predictions over the in-context set and frozen before the test set is touched.
  Max-F1 uses test-tuned thresholds and is reported only as an optimistic upper
  bound.
- **Intervals.** 2,000 bootstrap resamples of the test peptides, with all
  transforms scored on identical resamples and seeds averaged *inside* each
  resample. A difference is called significant when its two-sided 95% percentile
  interval excludes zero.
- **Scale.** Average precision, macro averages, per-label differences, bootstrap
  bounds and threshold-dependent scores are on the 0–100 scale. Log-likelihood
  (nats/peptide), MRR (0–1) and calibration error keep their natural scale.

## Hardware

Every number in the paper was produced on two NVIDIA RTX PRO 5000 Blackwell GPUs
(48 GB each) with PyTorch 2.13.0+cu130, scikit-learn 1.7.2, NumPy 2.2.6 and
Python 3.10.20. Smaller cards have not been tested; if inference runs out of
memory, lower `--chunk_size`. Post-hoc analyses that read the prediction cache
run on CPU in minutes.

## Not included

- **Feature extraction from raw sequence.** The computed matrices ship in
  `data/final/`, so this is not needed to reproduce anything. Of the extraction
  code itself, only `scripts/regen_ctd_fixed.py` (the 147 CTD columns) is here;
  the ESM-2, AAindex, physicochemical, motif and density blocks came from a
  separate pipeline. [docs/DATA.md](docs/DATA.md) documents every descriptor
  definition.
- **Manuscript figure scripts.** Figures 1, 2 and 5 are plotted from the values
  in the tables that `scripts/` produces.
- Exploratory analyses that did not enter the manuscript.

## Citation

_The preprint reference will be added here once the bioRxiv link is available._
See [`CITATION.cff`](CITATION.cff).

Please also cite the ESCAPE benchmark and TabPFN if you use this code.

## License

MIT — see [LICENSE](LICENSE). `external/escape/` is third-party code and is
**not** covered by it; see [`external/escape/NOTICE.md`](external/escape/NOTICE.md).
