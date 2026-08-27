# Paper → code map

Every table, figure and quantitative claim in the manuscript, with the script
that produces it and the artefact it lands in. Tables are named by their caption
rather than by number so this file survives manuscript renumbering.

All paths are relative to the repository root. Nothing under `results/` is
tracked; every script creates its own output directory and none overwrites an
existing file.

---

## Prerequisite: the prediction cache

`scripts/evaluate.py` fits all five transforms for all three seeds and writes
`data/preds_prob/<transform>/<transform>_seed<seed>.npy` — a dict holding the
out-of-fold training probabilities `(65870, 5)`, the test probabilities
`(16489, 5)`, and a SHA-256 fingerprint over both.

Everything in the "cache-only" column below reads that cache through
`src/data/preds_cache.py` instead of re-fitting, which is what keeps the
analyses mutually consistent and cheap. **Run Stage 1 first.**

---

## Methods (§2)

| Manuscript element | Script / module | Output |
|---|---|---|
| §2.1 Dataset, "Label structure of the ESCAPE benchmark" | — (benchmark as released) | `data/final/*.parquet`, tracked in this repository |
| §2.3 "Feature representation (330 numeric columns)" | matrices ship in `data/final/`; `scripts/regen_ctd_fixed.py` regenerates the 147 CTD columns, and `docs/DATA.md` defines the other seven families | `data/final/`, `data/final_v2/` |
| §2.5 The five multi-label transforms | `src/models/{binary_relevance,label_powerset,classifier_chain,ensemble_cc,probabilistic_cc}.py` | — |
| §2.2 Metrics, bootstrap convention | `src/utils/metrics.py` | — |
| §2.4 Backbone, "fit" vs "inference" | `src/models/base.py` | — |

---

## §3.1 Improved average precision across all activity labels

| Manuscript element | Script | GPU | Output |
|---|---|---|---|
| "Average precision (%) on the ESCAPE test set" — the frontier table, *This work* rows | `scripts/evaluate.py` | yes | `results/benchmark_all_seeds.json`, `results/benchmark_summary.csv`, `data/preds_prob/` |
| Per-seed values behind that table (Supplementary "Per-seed benchmark of all five multi-label transforms") | `scripts/collate_benchmark.py` | cache-only | `results/benchmark_all_seeds.json`, `results/benchmark_summary.csv` |
| Bootstrap intervals; PCC's Antiparasitic interval `[26.26, 48.50]`; per-label significance against published point estimates | `scripts/paired_ci_analysis.py` | cache-only | `results/analysis_v2/paired_ci/` |
| "Computational cost of each transform" — fit and inference seconds | `scripts/evaluate.py` (timings recorded per run) | yes | `results/benchmark_all_seeds.json` |
| Figure: five-label and powered macro AP (`fig1_frontier`) | plotted from the frontier table | — | — |
| Figure: per-label AP, published vs this work (`fig2_perlabel`) | plotted from the frontier table + `paired_ci` intervals | — | — |
| Published-baseline rows and the provenance table (Supplementary "Provenance of the published baselines") | as reported by the ESCAPE authors — no code here | — | — |

---

## §3.2 Separating data-size effects from model gains

| Manuscript element | Script | GPU | Output |
|---|---|---|---|
| Which released map folder holds which modality (contact vs distance maps) | `scripts/escape_inference.py --mode modality_check` | yes | `results/analysis_v2/escape_repro/modality_check.json` |
| Re-running the released ESCAPE checkpoints over the 5,495 covered test peptides; reproduction fidelity (macro AP 72.18 vs published 72.12) | `scripts/escape_inference.py --mode test_subset` | yes | `results/analysis_v2/escape_repro/escape_test_subset.json` |
| "Paired comparison against ESCAPE on the 5,495 test peptides" — Δ full-data column, pooled paired margins | `scripts/escape_paired_compare.py` | cache-only | `results/analysis_v2/escape_repro/` |
| Δ fold-matched column, its 95% CIs, and the direct full-data-minus-fold-matched measurement | `scripts/fold_matched_protocol.py` | yes | `results/analysis_v2/fold_matched/` |
| Checkpoint-averaging gains (+5.86 for ESCAPE vs +2.32 / +2.27 / +1.61 for LP / BR / CC) | `scripts/fold_matched_protocol.py` | yes | `results/analysis_v2/fold_matched/` |
| Half-context cost of 5.01 / 4.71 / 4.43 points motivating the experiment | `scripts/scaling_analysis.py` | yes | `results/scaling_benchmark/` |

Requires the ESCAPE checkpoints and structural maps — see `docs/DATA.md`.

---

## §3.3 Remote-homologue peptides show strongest gains

| Manuscript element | Script | GPU | Output |
|---|---|---|---|
| MMseqs2 search of test against in-context set; max identity per test peptide; the 32.4% ≥90% figure | `scripts/homology_analysis.py` | no (needs MMseqs2) | `results/analysis_v2/homology/` |
| "Homology stratification of the predefined test split" — per-band margins | `scripts/escape_paired_compare.py` (stratifies by the bands above) | cache-only | `results/analysis_v2/escape_repro/` |

`homology_analysis.py --keep_tsv` also retains the raw MMseqs2 hit table
(~167 MB), which `scripts/pu_next_assay.py` and the per-label identity joins
reuse.

---

## §3.4 Sequence-only information is sufficient

| Manuscript element | Script | GPU | Output |
|---|---|---|---|
| "Feature-group attribution on the stratified 4,000-peptide subsample" — group-only, leave-one-out and permutation columns | `scripts/feature_group_attribution.py` | yes | `results/analysis_v2/feature_attribution/` |
| Structural-branch ablation: zero / uniform-noise / swapped maps, max change 0.056 macro AP, 2.1e-5 max probability shift | `scripts/escape_structure_ablation.py` | yes | `results/analysis_v2/escape_repro/` |

---

## §3.5 Dependence helps mainly for scarce activities

| Manuscript element | Script | GPU | Output |
|---|---|---|---|
| Oracle conditional-predictability ceiling (mAP-4 76.73 vs BR floor 71.74); per-transform recovery percentages; per-activity oracle lift (`fig5_dependence`) | `scripts/oracle_dependence_ceiling.py` | yes | `results/label_dependence/oracle/` |
| Joint vs product held-out log-likelihood (−0.28266 vs −0.30271, +0.0200 nats/peptide) | `scripts/joint_likelihood_dependence.py` | yes | `results/label_dependence/joint_likelihood/` |
| CC soft vs hard chain context (69.70 → 72.43 mAP-4) | `scripts/cc_context_ablation.py` | yes | `results/analysis_v2/cc_context/` |
| CC-hard oracle recovery (−40.8% → +13.7%) | `scripts/cc_hard_followups.py` | cache-only | `results/analysis_v2/cc_hard_followups/` |
| "OR-gate violations" table — 0.00% (LP), 11.33% (PCC), 34.75% (BR), 39.51% (ECC), 44.85% (CC) over 246,000 predictions | `scripts/orgate_analysis.py` | cache-only | `results/analysis_v2/orgate/` |
| PCC's Antiparasitic gain over BR (+2.91, `[+0.81, +5.29]`) | `scripts/paired_ci_analysis.py` | cache-only | `results/analysis_v2/paired_ci/` |

---

## §3.6 Ranking activities from partial positive evidence

| Manuscript element | Script | GPU | Output |
|---|---|---|---|
| "Assay prioritization" table — Hit@1, MRR and pooled PU-AP at m = 0, 1, 2; eligible counts 4,235 / 1,035 / 250; permuted-label control | `scripts/pu_next_assay.py` | cache-only | `results/analysis_v2/pu_next_assay/` |
| Hit@1 decomposed by which activity is concealed (97.77% antibacterial … 3.03% antiparasitic) | `scripts/pu_next_assay.py` | cache-only | `results/analysis_v2/pu_next_assay/` |
| Conditional-posterior slicing, renormalisation and marginalisation | `src/utils/prioritization.py` | — | — |

---

## Supplementary

| Manuscript element | Script | GPU | Output |
|---|---|---|---|
| "Per-seed benchmark of all five multi-label transforms" — AP, threshold metrics, wall-clock, per seed | `scripts/evaluate.py` → `scripts/collate_benchmark.py` | yes → cache-only | `results/benchmark_all_seeds.json` |
| "Context-size scaling sweep" — n ∈ {250, 1250, 6250, 31250, 65870} for BR / LP / CC | `scripts/scaling_analysis.py --strategy all` | yes | `results/scaling_benchmark/` |
| "The multi-label transforms, with a worked example" | `src/models/` (the worked example is illustrative, no code) | — | — |
| "Provenance of the published baselines" | as reported by the original publications — no code here | — | — |

---

## Utility scripts

| Script | Purpose |
|---|---|
| `scripts/train.py` | Fit one transform for one seed and log to MLflow. The smallest useful entry point. |
| `scripts/collate_benchmark.py` | Rebuild the benchmark tables from the cached probabilities without refitting. |
| `scripts/regen_ctd_fixed.py` | Regenerate the 147 CTD columns with corrected amino-acid group definitions. |

---

## Runtime expectations

Order of magnitude on one 48 GB GPU, full 65,870-peptide context:

| Stage | Cost |
|---|---|
| `evaluate.py`, all five transforms × 3 seeds | hours — PCC alone is ~4,100 s of inference per seed, ECC ~3,800 s |
| `fold_matched_protocol.py`, `feature_group_attribution.py` | hours |
| `oracle_dependence_ceiling.py`, `cc_context_ablation.py`, `joint_likelihood_dependence.py` | tens of minutes to hours |
| `escape_inference.py`, `escape_structure_ablation.py` | minutes (small transformer, 5,495 peptides) |
| `homology_analysis.py` | minutes, CPU, dominated by MMseqs2 |
| `paired_ci_analysis.py`, `orgate_analysis.py`, `cc_hard_followups.py`, `pu_next_assay.py`, `collate_benchmark.py` | minutes, CPU, cache-only |
