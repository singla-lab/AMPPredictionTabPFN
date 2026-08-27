"""
Train/test sequence-homology audit and leakage-stratified re-evaluation.

Exact deduplication is already clean: Hash is sha256(Sequence)[:16] and there is
zero exact sequence overlap between fold1, fold2 and the test split. That rules
out the crudest failure mode but says nothing about near-duplicates — a test
peptide differing from a training peptide by one residue is still, for practical
purposes, memorised rather than predicted.

This script measures the real thing. MMseqs2 searches every test sequence
against the full training set; each test peptide is then labelled with the
maximum sequence identity it achieves against any training peptide. The cached
test probabilities are re-scored inside identity strata, so a model's accuracy
on genuinely novel peptides can be read off directly.

No refitting is involved: stratified evaluation is pure post-processing of the
existing predictions, so the numbers are directly comparable to the headline
benchmark.

Writes to results/analysis_v2/homology/ — creates new files, overwrites nothing.

Usage:
    python scripts/homology_analysis.py
    python scripts/homology_analysis.py --sensitivity 7.5 --min_qcov 0.5
"""
import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS, read_table
from src.data.preds_cache import (
    ANALYSIS_LABELS,
    SEEDS,
    STRATEGIES,
    label_indices,
    load_test_probs,
)
from src.utils import RunLogger
from src.utils.metrics import compute_paired_bootstrap_delta, fast_ap

DATA_DIR = REPO_ROOT / "data" / "final"
OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "homology"

CHILD_IDX = label_indices(ANALYSIS_LABELS)

# Identity bins. The <0.3 and <0.4 strata are the ones that matter: they are the
# peptides a model cannot have seen a close relative of during training.
BINS = [0.0, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0001]
BIN_LABELS = ["<30%", "30-40%", "40-50%", "50-70%", "70-90%", ">=90%"]


def write_fasta(seqs, ids, path):
    with open(path, "w") as f:
        for i, s in zip(ids, seqs):
            f.write(f">{i}\n{s}\n")


def run_mmseqs(query_fa, target_fa, out_tsv, tmp_dir, sensitivity, threads):
    """
    All-vs-all search of query against target.

    -s 7.5 is MMseqs2's most sensitive preset and --max-seqs is raised well above
    the default because short peptides generate many weak hits; a permissive
    -e keeps borderline alignments so that max-identity is not truncated by an
    arbitrary significance cutoff.
    """
    cmd = [
        "mmseqs", "easy-search", str(query_fa), str(target_fa),
        str(out_tsv), str(tmp_dir),
        "-s", str(sensitivity),
        "-e", "10000",
        "--max-seqs", "4000",
        "--alignment-mode", "3",
        "--threads", str(threads),
        "--format-output",
        "query,target,fident,alnlen,qcov,tcov,evalue,bits",
    ]
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"mmseqs failed with code {proc.returncode}")
    return out_tsv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensitivity", type=float, default=7.5)
    ap.add_argument("--min_qcov", type=float, default=0.5,
                    help="Minimum query coverage for a hit to count toward "
                         "max identity (a high-identity sliver of alignment is "
                         "not homology)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n_bootstraps", type=int, default=2000)
    ap.add_argument("--keep_tsv", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.which("mmseqs") is None:
        raise RuntimeError("mmseqs not found on PATH. conda install -c bioconda mmseqs2")

    print("Loading sequences ...", flush=True)
    tr = pd.concat([
        read_table(DATA_DIR / "final_fold1.csv", columns=["Hash", "Sequence"]),
        read_table(DATA_DIR / "final_fold2.csv", columns=["Hash", "Sequence"]),
    ], ignore_index=True)
    te = read_table(DATA_DIR / "final_test.csv",
                    columns=["Hash", "Sequence"] + LABEL_COLS)
    print(f"  train {len(tr)}   test {len(te)}", flush=True)

    exact = len(set(tr.Sequence) & set(te.Sequence))
    print(f"  exact sequence overlap: {exact}", flush=True)

    # ── MMseqs2 search ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        q_fa, t_fa = td / "test.fasta", td / "train.fasta"
        write_fasta(te.Sequence, te.Hash, q_fa)
        write_fasta(tr.Sequence, tr.Hash, t_fa)

        out_tsv = td / "hits.tsv"
        print("\nRunning MMseqs2 (test vs train) ...", flush=True)
        run_mmseqs(q_fa, t_fa, out_tsv, td / "tmp", args.sensitivity, args.threads)

        hits = pd.read_csv(out_tsv, sep="\t", header=None, names=[
            "query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"])
        if args.keep_tsv:
            shutil.copy(out_tsv, OUT_DIR / "mmseqs_hits.tsv")

    print(f"  {len(hits):,} raw hits for {hits['query'].nunique():,} test peptides",
          flush=True)

    # ── max identity per test peptide ─────────────────────────────────────────
    covered = hits[hits.qcov >= args.min_qcov]
    max_id_cov = covered.groupby("query")["fident"].max()
    max_id_any = hits.groupby("query")["fident"].max()

    te["max_identity"] = te.Hash.map(max_id_cov).fillna(0.0)
    te["max_identity_anycov"] = te.Hash.map(max_id_any).fillna(0.0)
    te["stratum"] = pd.cut(te.max_identity, bins=BINS, labels=BIN_LABELS,
                           right=False, include_lowest=True)

    print("\n--- Max identity of each test peptide vs the training set ---")
    print(f"  (query coverage >= {args.min_qcov})")
    desc = te.max_identity.describe(percentiles=[.05, .25, .5, .75, .95])
    print(desc.round(4).to_string())
    n_ident = int((te.max_identity >= 0.99).sum())
    print(f"\n  test peptides with >=99% identity to a training peptide: "
          f"{n_ident} ({100*n_ident/len(te):.2f}%)")
    print(f"  test peptides with  <40% identity                        : "
          f"{int((te.max_identity < 0.4).sum())} "
          f"({100*(te.max_identity < 0.4).mean():.2f}%)")

    strat_counts = te.groupby("stratum", observed=False).size()
    print("\n--- Test peptides per identity stratum ---")
    lab_counts = te.groupby("stratum", observed=False)[LABEL_COLS].sum()
    tbl = pd.concat([strat_counts.rename("n_peptides"), lab_counts], axis=1)
    print(tbl.to_string())

    # ── stratified re-scoring of cached predictions ───────────────────────────
    Y_test = te[LABEL_COLS].values.astype(np.int32)
    test_probs = load_test_probs(STRATEGIES, SEEDS)

    rows = []
    for stratum in BIN_LABELS + ["<40% (cumulative)", "<30% (cumulative)", "ALL"]:
        if stratum == "ALL":
            mask = np.ones(len(te), dtype=bool)
        elif stratum == "<40% (cumulative)":
            mask = (te.max_identity < 0.4).values
        elif stratum == "<30% (cumulative)":
            mask = (te.max_identity < 0.3).values
        else:
            mask = (te.stratum == stratum).values
        n = int(mask.sum())
        if n < 30:
            continue
        Ys = Y_test[mask]
        for strat in STRATEGIES:
            per_label = {}
            for j, lbl in enumerate(LABEL_COLS):
                if Ys[:, j].sum() == 0 or Ys[:, j].sum() == n:
                    per_label[lbl] = float("nan")   # AP undefined in this stratum
                else:
                    per_label[lbl] = float(np.mean(
                        [fast_ap(Ys[:, j], p[mask, j]) for p in test_probs[strat]]))
            rec = {"stratum": stratum, "n_peptides": n, "Strategy": strat}
            rec.update({f"AP_{k}": v for k, v in per_label.items()})
            rec["mAP5"] = float(np.nanmean([per_label[l] for l in LABEL_COLS]))
            rec["mAP4"] = float(np.nanmean([per_label[l] for l in ANALYSIS_LABELS]))
            rows.append(rec)

    df = pd.DataFrame(rows)
    print("\n--- mAP by identity stratum (mean over 3 seeds) ---")
    piv = df.pivot(index="stratum", columns="Strategy", values="mAP5")
    order = [s for s in BIN_LABELS + ["<40% (cumulative)", "<30% (cumulative)", "ALL"]
             if s in piv.index]
    npep = df.drop_duplicates("stratum").set_index("stratum")["n_peptides"]
    piv = piv.reindex(order)
    piv.insert(0, "n", npep.reindex(order))
    print(piv.round(4).to_string())

    print("\n--- mAP-4 (activities only) by identity stratum ---")
    piv4 = df.pivot(index="stratum", columns="Strategy", values="mAP4").reindex(order)
    piv4.insert(0, "n", npep.reindex(order))
    print(piv4.round(4).to_string())

    # ── does the LP/PCC advantage survive on novel peptides? ──────────────────
    print("\n--- Paired bootstrap vs BR within the <40% identity stratum ---",
          flush=True)
    mask = (te.max_identity < 0.4).values
    boot = []
    if mask.sum() >= 100:
        for strat in ["LP", "PCC", "CC", "ECC"]:
            for tag, cols in (("mAP5", list(range(len(LABEL_COLS)))),
                              ("mAP4", CHILD_IDX)):
                r = compute_paired_bootstrap_delta(
                    Y_test[mask][:, cols],
                    [p[mask][:, cols] for p in test_probs["BR"]],
                    [p[mask][:, cols] for p in test_probs[strat]],
                    n_bootstraps=args.n_bootstraps,
                )
                boot.append({"comparison": f"{strat} - BR", "scope": tag,
                             "stratum": "<40%", **r})
        df_boot = pd.DataFrame(boot)
        show = df_boot[["comparison", "scope", "delta", "ci_lower",
                        "ci_upper", "significant"]].copy()
        for c in ["delta", "ci_lower", "ci_upper"]:
            show[c] = show[c].map(lambda v: f"{v:+.4f}")
        print(show.to_string(index=False))
    else:
        df_boot = pd.DataFrame()
        print("  too few peptides below 40% identity for a stable bootstrap")

    # ── save ──────────────────────────────────────────────────────────────────
    te[["Hash", "max_identity", "max_identity_anycov", "stratum"]].to_csv(
        OUT_DIR / "test_max_identity.csv", index=False)
    df.to_csv(OUT_DIR / "stratified_ap.csv", index=False)
    tbl.to_csv(OUT_DIR / "stratum_label_counts.csv")
    if len(df_boot):
        df_boot.to_csv(OUT_DIR / "hard_subset_paired_bootstrap.csv", index=False)

    (OUT_DIR / "homology_summary.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(),
        "generated_by": "scripts/homology_analysis.py",
        "mmseqs_version": subprocess.run(["mmseqs", "version"],
                                         capture_output=True, text=True).stdout.strip(),
        "sensitivity": args.sensitivity,
        "min_qcov": args.min_qcov,
        "exact_sequence_overlap": exact,
        "n_test": len(te),
        "n_train": len(tr),
        "identity_describe": desc.to_dict(),
        "n_test_ge99_identity": n_ident,
        "n_test_lt40_identity": int((te.max_identity < 0.4).sum()),
        "n_test_lt30_identity": int((te.max_identity < 0.3).sum()),
        "stratum_counts": strat_counts.to_dict(),
    }, indent=2, default=str))

    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    with RunLogger(
        script_name="homology_analysis.py",
        log_dir=REPO_ROOT / "logs",
        sample_interval_s=30.0,
        extra_meta={"argv": sys.argv},
    ):
        main()
