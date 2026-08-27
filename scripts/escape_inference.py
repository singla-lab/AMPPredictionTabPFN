"""
Run the released ESCAPE checkpoints over the peptides whose structural maps the
authors published, to obtain per-peptide predictions for a paired comparison.

Scope and honesty constraints
-----------------------------
The ESCAPE Drive release contains two .npy folders:

  distance_maps_Test  5,495 unique hashes, ALL of them in our test split
                      (33.3% of our 16,489 test peptides), int64 0/1 -> these
                      are binary CONTACT maps despite the folder name.
  Distance_Maps       5,499 unique hashes spread across fold1 (2,191),
                      fold2 (2,179), test (1,044) and 85 unmatched, float32 in
                      angstrom units -> real-valued DISTANCE maps.

Neither folder covers a full split, so ESCAPE's published headline number over
the whole test set CANNOT be reproduced. What is obtainable is the ensemble's
per-peptide predictions on the 5,495-peptide subset it does cover, which is
enough for a paired comparison against our strategies re-scored on exactly the
same peptides.

Because the two folders disagree on modality, and 1,044 test peptides appear in
BOTH, `--mode modality_check` scores that overlap under each modality with the
same checkpoints. Whichever modality the checkpoints were actually trained on
should score far better; this is decided by measurement, not by folder name.

Normalisation caveat: ESCAPE's test_ESCAPE.py derives global_min/global_max by
globbing every .npy in maps_dir, so the constants depend on which files are
present. With only a subset available the constants cannot match the authors'.
They are recomputed over whatever set is being scored and recorded in the
output JSON so the deviation is explicit. For 0/1 contact maps the transform is
the identity, so the caveat only bites for the float32 distance maps.

Writes to results/analysis_v2/escape_repro/. Nothing under data/preds_prob/ or
data/final/ is touched.

Usage:
    python scripts/escape_inference.py --mode modality_check --device cuda:0
    python scripts/escape_inference.py --mode test_subset   --device cuda:0
"""
import argparse
import datetime
import glob
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external.escape.models import MultiModalClassifier
from src.data import read_table
from src.utils.metrics import fast_ap

ESCAPE_DIR = REPO_ROOT / "data" / "escape_repro"
OUT_DIR = REPO_ROOT / "results" / "analysis_v2" / "escape_repro"

# ESCAPE's own label order (external/escape/dataset.py: LABEL_COLUMNS). The
# checkpoints' output units are in THIS order; our LABEL_COLS may differ, so
# every downstream join is by name, never by position.
ESCAPE_LABELS = ["Antibacterial", "Antifungal", "Antiviral", "Antiparasitic",
                 "Antimicrobial"]

AA_LIST = ['-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
VOCAB = {aa: i + 1 for i, aa in enumerate(AA_LIST)}


class SubsetDataset(Dataset):
    """
    ESCAPEDataset restricted to rows whose map file actually exists.

    Identical tokenisation, normalisation and resizing to
    external/escape/dataset.py; the only change is that the row list is filtered
    up front rather than letting np.load raise on a missing file.
    """

    def __init__(self, df, maps_dir, seq_max_len, global_min, global_max, img_size):
        self.df = df.reset_index(drop=True)
        self.maps_dir = maps_dir
        self.seq_max_len = seq_max_len
        self.global_min = global_min
        self.global_max = global_max
        self.img_size = img_size

    def seq_to_ids(self, seq):
        seq = str(seq).upper()
        ids = [VOCAB.get(c, 0) for c in seq[:self.seq_max_len]]
        ids += [0] * (self.seq_max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_ids = self.seq_to_ids(row["Sequence"])
        mat = np.load(os.path.join(self.maps_dir, f"{row['Hash']}.npy")).astype(np.float32)
        mat = (mat - self.global_min) / (self.global_max - self.global_min + 1e-8)
        mat = np.clip(mat, 0.0, 1.0)
        x = torch.from_numpy(mat).unsqueeze(0)
        x = F.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False).squeeze(0)
        labels = torch.tensor(row[ESCAPE_LABELS].values.astype(np.float32),
                              dtype=torch.float32)
        return seq_ids, x, labels


def global_range(maps_dir, hashes):
    """ESCAPE's global_min/global_max, over exactly the files being scored."""
    lo, hi = float("inf"), -float("inf")
    for h in hashes:
        m = np.load(os.path.join(maps_dir, f"{h}.npy"))
        lo = min(lo, float(m.min()))
        hi = max(hi, float(m.max()))
    return lo, hi


def build_model(device):
    return MultiModalClassifier(
        seq_d_model=256, struct_d_model=192, n_heads=8, num_layers=4,
        num_classes=5, vocab_size=27, max_len_seq=200,
        img_size=224, patch_size=16, img_channels=1,
    ).to(device)


@torch.no_grad()
def run_checkpoint(ckpt, loader, device):
    model = build_model(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    probs, labels = [], []
    for seq_ids, dmap, y in loader:
        out = model(seq_ids.to(device), dmap.to(device))
        probs.append(torch.sigmoid(out).cpu().numpy())
        labels.append(y.numpy())
    del model
    torch.cuda.empty_cache()
    return np.vstack(labels), np.vstack(probs)


def metrics(y_true, y_probs):
    """ESCAPE's own metric definitions (test_ESCAPE.py compute_metrics)."""
    from sklearn.metrics import precision_recall_curve
    aps, f1s = [], []
    for i in range(y_true.shape[1]):
        aps.append(float(fast_ap(y_true[:, i], y_probs[:, i])))
        p, r, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1s.append(float(np.max(2 * p * r / (p + r + 1e-8))))
    return aps, f1s, float(np.mean(aps)), float(np.mean(f1s))


def score(df, maps_dir, args, device, tag):
    hashes = df["Hash"].astype(str).tolist()
    lo, hi = global_range(maps_dir, hashes)
    ds = SubsetDataset(df, maps_dir, args.seq_max_len, lo, hi, args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    c1 = ESCAPE_DIR / "ckpt" / "Best_model_Fold1.pth"
    c2 = ESCAPE_DIR / "ckpt" / "Best_model_Fold2.pth"
    y1, p1 = run_checkpoint(c1, loader, device)
    y2, p2 = run_checkpoint(c2, loader, device)
    assert np.array_equal(y1, y2), "label mismatch between checkpoint passes"
    p_ens = (p1 + p2) / 2.0

    out = {"tag": tag, "n": int(len(df)), "maps_dir": str(maps_dir),
           "global_min": lo, "global_max": hi, "labels": ESCAPE_LABELS}
    for name, p in [("fold1", p1), ("fold2", p2), ("ensemble", p_ens)]:
        aps, f1s, mAP, mF1 = metrics(y1, p)
        out[name] = {"AP": dict(zip(ESCAPE_LABELS, aps)),
                     "F1": dict(zip(ESCAPE_LABELS, f1s)),
                     "macro_AP": mAP, "macro_F1": mF1}
        print(f"  [{tag}/{name}] macro_AP={mAP:.4f} macro_F1={mF1:.4f}  " +
              "  ".join(f"{l}={a:.3f}" for l, a in zip(ESCAPE_LABELS, aps)), flush=True)
    return out, y1, p1, p2, p_ens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modality_check", "test_subset"], required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_max_len", type=int, default=200)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    test = read_table(REPO_ROOT / "data" / "final" / "final_test.csv")
    test["Hash"] = test["Hash"].astype(str)
    for c in ESCAPE_LABELS:
        assert c in test.columns, f"label {c} missing from final_test.csv"

    have_contact = {os.path.basename(p)[:-4]
                    for p in glob.glob(str(ESCAPE_DIR / "maps_test" / "*.npy"))}
    have_dist = {os.path.basename(p)[:-4]
                 for p in glob.glob(str(ESCAPE_DIR / "maps_main" / "*.npy"))}
    print(f"contact maps on disk: {len(have_contact):,}   "
          f"distance maps on disk: {len(have_dist):,}", flush=True)

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    if args.mode == "modality_check":
        both = sorted(have_contact & have_dist)
        res = {"generated": stamp, "n_overlap_on_disk": len(both)}
        if both:
            # Preferred form: same peptides under both modalities, so the
            # comparison is paired and peptide difficulty cancels.
            df_c = df_d = test[test["Hash"].isin(both)].copy()
            res["design"] = "paired"
            print(f"PAIRED modality check on {len(df_c):,} peptides in BOTH folders",
                  flush=True)
        else:
            # Fallback: the Drive quota stopped us before any hash was fetched
            # under both modalities. Both arms are still drawn from the same
            # test split, so the comparison is UNPAIRED and peptide difficulty
            # does not cancel -- only a large margin is interpretable.
            df_c = test[test["Hash"].isin(have_contact)].copy()
            df_d = test[test["Hash"].isin(have_dist)].copy()
            res["design"] = "unpaired"
            print(f"UNPAIRED modality check: {len(df_c):,} contact vs "
                  f"{len(df_d):,} distance peptides (disjoint sets)", flush=True)
        for tag, d, dd in [("contact", ESCAPE_DIR / "maps_test", df_c),
                           ("distance", ESCAPE_DIR / "maps_main", df_d)]:
            r, _, _, _, _ = score(dd, str(d), args, device, tag)
            res[tag] = r
        c = res["contact"]["ensemble"]["macro_AP"]
        d = res["distance"]["ensemble"]["macro_AP"]
        res["verdict"] = ("contact" if c > d else "distance")
        res["margin_macro_AP"] = abs(c - d)
        print(f"\nVERDICT ({res['design']}): checkpoints score higher on "
              f"{res['verdict']} maps ({max(c, d):.4f} vs {min(c, d):.4f})", flush=True)
        (OUT_DIR / "modality_check.json").write_text(json.dumps(res, indent=2))

    else:
        df = test[test["Hash"].isin(have_contact)].copy()
        print(f"scoring {len(df):,} / {len(test):,} test peptides "
              f"({100 * len(df) / len(test):.1f}%)", flush=True)
        r, y, p1, p2, pe = score(df, str(ESCAPE_DIR / "maps_test"), args, device,
                                 "test_subset")
        r["generated"] = stamp
        r["frac_of_test"] = float(len(df) / len(test))
        (OUT_DIR / "escape_test_subset.json").write_text(json.dumps(r, indent=2))
        # Per-peptide predictions, keyed by Hash so downstream joins are by name.
        np.savez_compressed(
            OUT_DIR / "escape_test_subset_preds.npz",
            hashes=df["Hash"].values.astype(str),
            labels=np.array(ESCAPE_LABELS),
            y_true=y, fold1=p1, fold2=p2, ensemble=pe,
        )
        print(f"wrote {OUT_DIR/'escape_test_subset_preds.npz'}", flush=True)


if __name__ == "__main__":
    main()
