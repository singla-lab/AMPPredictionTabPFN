"""
Per-peptide paired comparison against the reproduced ESCAPE ensemble.

Everything here is restricted to the peptides ESCAPE's released structural maps
cover, so both sides are evaluated on an identical peptide set and the paired
bootstrap resamples the same peptides for both arms. This is the comparison the
published-number table could not make: that table set our full-test-set AP
against ESCAPE's reported AP, two different peptide sets and no shared
resampling, so no interval on the difference was available.

Steps
  1. Reproduction fidelity: our re-run of the released checkpoints vs the values
     the ESCAPE paper reports, per label.
  2. Re-score BR/LP/CC/ECC/PCC on exactly the covered subset, seed-averaged.
  3. Paired bootstrap on (ours - ESCAPE) per label and on macro mAP-5/mAP-4.
  4. Homology stratification: the same delta within each identity band from the
     MMseqs2 easy-search analysis, to test whether any advantage is confined to
     peptides with close training homologues.

Alignment is by label NAME and by peptide Hash throughout; both are asserted.

Writes to results/analysis_v2/escape_repro/. Reads only.

Usage:
    python scripts/escape_paired_compare.py
"""
import datetime
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS, read_table
from src.data.preds_cache import ANALYSIS_LABELS, SEEDS, load_test_probs
from src.utils.metrics import compute_paired_bootstrap_delta, fast_ap

OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "escape_repro"
STRATEGIES = ["BR", "LP", "CC", "ECC", "PCC"]

# As printed in the ESCAPE paper, per label.
ESCAPE_PUBLISHED = {"Antibacterial": 0.942, "Antifungal": 0.634, "Antiviral": 0.698,
                    "Antiparasitic": 0.376, "Antimicrobial": 0.956}
PUBLISHED_MACRO = 0.7212


def paired_delta_ap(y, pa, pb, n_boot=2000, seed=42):
    """Bootstrap CI for AP(b) - AP(a) on shared resamples of the same peptides."""
    rng = np.random.RandomState(seed)
    n = len(y)
    obs = fast_ap(y, pb) - fast_ap(y, pa)
    d = np.empty(n_boot)
    ok = 0
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            d[ok] = np.nan
            continue
        d[ok] = fast_ap(y[idx], pb[idx]) - fast_ap(y[idx], pa[idx])
        ok += 1
    d = d[:ok]
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(obs), "ci_lower": float(lo), "ci_upper": float(hi),
            "significant": bool(lo > 0 or hi < 0), "n_boot_used": int(len(d))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    z = np.load(OUT_DIR / "escape_test_subset_preds.npz", allow_pickle=True)
    e_hash = z["hashes"].astype(str)
    e_labels = list(z["labels"])
    e_prob = z["ensemble"]
    e_true = z["y_true"]

    test = read_table(REPO_ROOT / "data" / "final" / "final_test.csv")
    test["Hash"] = test["Hash"].astype(str)

    # --- align rows by Hash, columns by label NAME -----------------------------
    pos = {h: i for i, h in enumerate(test["Hash"].values)}
    assert len(pos) == len(test), "duplicate Hash in final_test.csv"
    rows = np.array([pos[h] for h in e_hash])
    assert (test["Hash"].values[rows] == e_hash).all(), "row alignment failed"

    col_of = {c: LABEL_COLS.index(c) for c in e_labels}
    Y = test.loc[rows, e_labels].values.astype(int)
    assert np.array_equal(Y, e_true.astype(int)), \
        "ground truth from final_test.csv disagrees with the labels ESCAPE saw"
    print(f"aligned {len(rows):,} peptides, labels {e_labels}", flush=True)

    # --- 1. reproduction fidelity ---------------------------------------------
    fid = []
    for j, lab in enumerate(e_labels):
        got = float(fast_ap(Y[:, j], e_prob[:, j]))
        fid.append({"label": lab, "reproduced_AP": got,
                    "published_AP": ESCAPE_PUBLISHED[lab],
                    "abs_diff": abs(got - ESCAPE_PUBLISHED[lab]),
                    "n_pos": int(Y[:, j].sum())})
    fid_df = pd.DataFrame(fid)
    macro_repro = float(fid_df["reproduced_AP"].mean())
    fid_df.to_csv(OUT_DIR / "reproduction_fidelity.csv", index=False)
    print(f"\n=== reproduction fidelity ({len(rows):,}-peptide subset vs published) ===")
    print(fid_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"macro reproduced={macro_repro:.4f}  published={PUBLISHED_MACRO:.4f}  "
          f"diff={abs(macro_repro-PUBLISHED_MACRO):.4f}")
    excl = fid_df[fid_df.label != "Antiparasitic"]
    print(f"excluding Antiparasitic: mean |diff| = {excl.abs_diff.mean():.4f} "
          f"(max {excl.abs_diff.max():.4f})")

    # --- 2. our strategies on the same peptides -------------------------------
    cache = load_test_probs(strategies=STRATEGIES)
    ours = {}
    for s in STRATEGIES:
        # seed-average first, then subset the rows ESCAPE covers
        p = np.mean(np.stack(cache[s]), axis=0)
        ours[s] = np.column_stack([p[rows, col_of[c]] for c in e_labels])
        assert ours[s].shape == e_prob.shape

    ap_rows = []
    for name, p in [("ESCAPE", e_prob)] + [(s, ours[s]) for s in STRATEGIES]:
        r = {"model": name}
        for j, lab in enumerate(e_labels):
            r[lab] = float(fast_ap(Y[:, j], p[:, j]))
        r["mAP-5"] = float(np.mean([r[l] for l in e_labels]))
        r["mAP-4"] = float(np.mean([r[l] for l in ANALYSIS_LABELS]))
        ap_rows.append(r)
    ap_df = pd.DataFrame(ap_rows)
    ap_df.to_csv(OUT_DIR / "subset_ap_table.csv", index=False)
    print(f"\n=== AP on the {len(rows):,} covered peptides (ours seed-averaged) ===")
    print(ap_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- 3. paired bootstrap, ours - ESCAPE -----------------------------------
    print("\n=== paired bootstrap: ours - ESCAPE (same peptides, shared resamples) ===")
    deltas = []
    for s in STRATEGIES:
        for j, lab in enumerate(e_labels):
            d = paired_delta_ap(Y[:, j], e_prob[:, j], ours[s][:, j])
            d.update({"strategy": s, "label": lab, "n_pos": int(Y[:, j].sum())})
            deltas.append(d)
        for macro_name, subset in [("mAP-5", e_labels), ("mAP-4", ANALYSIS_LABELS)]:
            cols = [e_labels.index(c) for c in subset]
            d = paired_macro(Y, e_prob, ours[s], cols)
            d.update({"strategy": s, "label": macro_name, "n_pos": -1})
            deltas.append(d)
    dd = pd.DataFrame(deltas)[["strategy", "label", "n_pos", "delta",
                               "ci_lower", "ci_upper", "significant"]]
    dd.to_csv(OUT_DIR / "paired_delta_vs_escape.csv", index=False)
    print(dd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- 4. homology stratification -------------------------------------------
    hom_path = REPO_ROOT / "results" / "analysis_v2" / "homology"
    strat_out = None
    band_file = hom_path / "test_max_identity.csv"
    if not band_file.exists():
        print(f"\n[homology] {band_file} missing -- stratification skipped", flush=True)
    else:
        hb = pd.read_csv(band_file)
        hb["Hash"] = hb["Hash"].astype(str)
        band_col = "stratum"
        m = hb.set_index("Hash")[band_col].reindex(e_hash)
        print(f"\n=== homology stratification ({band_file.name}, col '{band_col}') ===")
        print(m.value_counts(dropna=False).to_string())
        # A label with no positives inside a band has AP identically 0 for every
        # model, so including it forces a delta of exactly 0 and dilutes the
        # macro toward 0. Restrict each band's macro to labels that actually
        # have positives there, and record which ones were used.
        MIN_POS = 5
        srows = []
        for band, sel in m.groupby(m):
            mask = np.isin(e_hash, sel.index.values)
            if mask.sum() < 100:
                continue
            for s in STRATEGIES:
                for macro_name, subset in [("mAP-5", e_labels), ("mAP-4", ANALYSIS_LABELS)]:
                    cols = [e_labels.index(c) for c in subset
                            if Y[mask, e_labels.index(c)].sum() >= MIN_POS]
                    if not cols:
                        continue
                    d = paired_macro(Y[mask], e_prob[mask], ours[s][mask], cols)
                    d.update({"band": band, "n": int(mask.sum()),
                              "strategy": s, "metric": macro_name,
                              "labels_used": "|".join(e_labels[j] for j in cols),
                              "n_labels_used": len(cols)})
                    srows.append(d)
        strat_out = pd.DataFrame(srows)
        strat_out.to_csv(OUT_DIR / "paired_delta_by_homology.csv", index=False)
        print(strat_out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    json.dump({"generated": stamp, "n_peptides": int(len(rows)),
               "frac_of_test": float(len(rows) / len(test)),
               "macro_reproduced": macro_repro, "macro_published": PUBLISHED_MACRO,
               "labels": e_labels, "seeds": SEEDS},
              open(OUT_DIR / "paired_compare_meta.json", "w"), indent=2)
    print(f"\nwrote outputs to {OUT_DIR}")


def paired_macro(Y, pa, pb, cols, n_boot=2000, seed=42):
    """Bootstrap CI for macro-AP(b) - macro-AP(a) over the given label columns."""
    rng = np.random.RandomState(seed)
    n = len(Y)
    obs = (np.mean([fast_ap(Y[:, j], pb[:, j]) for j in cols])
           - np.mean([fast_ap(Y[:, j], pa[:, j]) for j in cols]))
    d = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            a = np.mean([fast_ap(Y[idx, j], pa[idx, j]) for j in cols])
            b = np.mean([fast_ap(Y[idx, j], pb[idx, j]) for j in cols])
        except Exception:
            continue
        if np.isfinite(a) and np.isfinite(b):
            d.append(b - a)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(obs), "ci_lower": float(lo), "ci_upper": float(hi),
            "significant": bool(lo > 0 or hi < 0), "n_boot_used": int(len(d))}


if __name__ == "__main__":
    main()
