"""
Does ESCAPE's structural branch affect ESCAPE's predictions?

Motivation. While reproducing ESCAPE (Section 20) the two published map folders
turned out to hold different modalities -- binary contact maps and real-valued
distance maps -- yet scoring the same peptides under either gave macro AP equal
to four decimal places. That coincidence is only possible if the structural input
barely reaches the output, so it is tested directly here.

Design. The sequence input is held fixed and only the structural input is
replaced, so any change in the output is attributable to the structural branch
alone. Four inputs are compared against the real maps:

  real       the peptide's own map (reference)
  zeros      an all-zero image
  random     uniform noise in [0, 1]
  shuffled   another peptide's real map (correct marginal statistics, wrong
             pairing) -- the strongest of the three controls, because it leaves
             the input distribution untouched and destroys only the association
             between a peptide and its own structure

Reported per checkpoint and for the ensemble: macro AP under each input, and the
distribution of |p_alt - p_real| over every peptide-label pair.

Scope. This measures the RELEASED CHECKPOINTS AT INFERENCE. It does not establish
what the structural branch contributed during training, where it could still have
acted as a regulariser.

Writes to results/analysis_v2/escape_repro/structure_ablation.{csv,json}. Reads only.

Usage:
    python scripts/escape_structure_ablation.py --device cuda:0
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

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from external.escape.models import MultiModalClassifier
from src.data import read_table
from src.utils.metrics import fast_ap

ESCAPE_DIR = REPO / "data" / "escape_repro"
OUT_DIR = REPO / "results" / "analysis_v2" / "escape_repro"
LBL = ["Antibacterial", "Antifungal", "Antiviral", "Antiparasitic", "Antimicrobial"]
AA = ['-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
      'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
VOCAB = {a: i + 1 for i, a in enumerate(AA)}


def tokenize(seq, n=200):
    ids = [VOCAB.get(c, 0) for c in str(seq).upper()[:n]]
    return ids + [0] * (n - len(ids))


def load_maps(hashes, maps_dir, img_size, lo, hi):
    out = torch.empty(len(hashes), 1, img_size, img_size)
    for i, h in enumerate(hashes):
        m = np.load(os.path.join(maps_dir, f"{h}.npy")).astype(np.float32)
        m = np.clip((m - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        t = torch.from_numpy(m)[None, None]
        out[i] = F.interpolate(t, size=(img_size, img_size), mode="bilinear",
                               align_corners=False)[0]
    return out


def build(device):
    return MultiModalClassifier(
        seq_d_model=256, struct_d_model=192, n_heads=8, num_layers=4,
        num_classes=5, vocab_size=27, max_len_seq=200, img_size=224,
        patch_size=16, img_channels=1).to(device)


@torch.no_grad()
def infer(model, X, M, device, bs=64):
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(
            model(X[i:i + bs].to(device), M[i:i + bs].to(device))).cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    maps_dir = str(ESCAPE_DIR / "maps_test")
    have = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{maps_dir}/*.npy"))
    test = read_table(REPO / "data" / "final" / "final_test.csv")
    test["Hash"] = test["Hash"].astype(str)
    df = test[test["Hash"].isin(set(have))].reset_index(drop=True)
    print(f"structural ablation on {len(df):,} peptides", flush=True)

    if not have:
        raise SystemExit(
            f"No structural maps found in {maps_dir}. This analysis needs the maps "
            f"released with the ESCAPE benchmark; see docs/DATA.md section 4."
        )

    lo = min(float(np.load(f"{maps_dir}/{h}.npy").min()) for h in have)
    hi = max(float(np.load(f"{maps_dir}/{h}.npy").max()) for h in have)
    print(f"global_min={lo:.4f} global_max={hi:.4f}", flush=True)

    X = torch.tensor([tokenize(s) for s in df.Sequence], dtype=torch.long)
    M = load_maps(df.Hash.values, maps_dir, args.img_size, lo, hi)
    Y = df[LBL].values.astype(int)

    g = torch.Generator().manual_seed(args.seed)
    variants = {
        "real": M,
        "zeros": torch.zeros_like(M),
        "random": torch.rand(M.shape, generator=g),
        "shuffled": M[torch.randperm(len(M), generator=g)],
    }

    rows, probs = [], {}
    for ck in ["Best_model_Fold1.pth", "Best_model_Fold2.pth"]:
        model = build(device)
        model.load_state_dict(torch.load(ESCAPE_DIR / "ckpt" / ck, map_location=device))
        model.eval()
        tag = ck.replace("Best_model_", "").replace(".pth", "")
        for name, mm in variants.items():
            p = infer(model, X, mm, device, args.batch_size)
            probs[(tag, name)] = p
            aps = [float(fast_ap(Y[:, j], p[:, j])) for j in range(5)]
            d = np.abs(p - probs[(tag, "real")])
            rows.append({"checkpoint": tag, "struct_input": name,
                         **{f"AP_{l}": a for l, a in zip(LBL, aps)},
                         "macro_AP": float(np.mean(aps)),
                         "max_abs_dprob": float(d.max()),
                         "mean_abs_dprob": float(d.mean()),
                         "frac_dprob_gt_0.01": float((d > 0.01).mean())})
            print(f"  [{tag}/{name:8s}] macroAP={np.mean(aps):.6f} "
                  f"maxΔ={d.max():.6f} meanΔ={d.mean():.8f}", flush=True)
        del model
        torch.cuda.empty_cache()

    for name in variants:
        p = (probs[("Fold1", name)] + probs[("Fold2", name)]) / 2.0
        probs[("ensemble", name)] = p
        aps = [float(fast_ap(Y[:, j], p[:, j])) for j in range(5)]
        d = np.abs(p - probs[("ensemble", "real")])
        rows.append({"checkpoint": "ensemble", "struct_input": name,
                     **{f"AP_{l}": a for l, a in zip(LBL, aps)},
                     "macro_AP": float(np.mean(aps)),
                     "max_abs_dprob": float(d.max()),
                     "mean_abs_dprob": float(d.mean()),
                     "frac_dprob_gt_0.01": float((d > 0.01).mean())})
        print(f"  [ensemble/{name:8s}] macroAP={np.mean(aps):.6f} "
              f"maxΔ={d.max():.6f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "structure_ablation.csv", index=False)

    ens = out[out.checkpoint == "ensemble"].set_index("struct_input")
    json.dump({
        "generated_at": datetime.datetime.now().isoformat(),
        "generated_by": "scripts/escape_structure_ablation.py",
        "n_peptides": int(len(df)), "global_min": lo, "global_max": hi,
        "scope": "released checkpoints at inference; says nothing about training",
        "ensemble_macro_AP": {k: float(ens.loc[k, "macro_AP"]) for k in variants},
        "max_macro_AP_spread": float(ens.macro_AP.max() - ens.macro_AP.min()),
    }, open(OUT_DIR / "structure_ablation.json", "w"), indent=2)

    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nensemble macro-AP spread across all four structural inputs: "
          f"{ens.macro_AP.max() - ens.macro_AP.min():.2e}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
