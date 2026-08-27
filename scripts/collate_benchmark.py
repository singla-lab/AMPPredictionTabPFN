"""
Collate existing probability caches and logs to regenerate benchmark_all_seeds.json
and benchmark_summary.csv without refitting any TabPFN models.
"""
import json
import pathlib
import sys
import re
import gc
import platform

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LABEL_COLS
from src.utils import (
    compute_all_metrics,
    compute_ap,
    compute_bootstrap_ci,
    compute_paired_statistical_tests,
)

from scripts.evaluate import load_train_test

SEEDS = [42, 1665, 8914]
STRATEGIES = ["BR", "LP", "CC", "ECC", "PCC"]
DATA_DIR    = REPO_ROOT / "data" / "final"
RESULTS_DIR = REPO_ROOT / "results"
PREDS_DIR   = REPO_ROOT / "data" / "preds_prob"
LOGS_DIR    = REPO_ROOT / "logs"

def parse_logs():
    """Extract timings from all evaluate_*.log files in logs/"""
    timings = {}
    for log_file in LOGS_DIR.glob("evaluate_*.log"):
        with open(log_file, "r") as f:
            current_strategy = None
            for line in f:
                strat_match = re.search(r"^\s*Strategy:\s*(\w+)", line)
                if strat_match:
                    current_strategy = strat_match.group(1)
                
                if current_strategy:
                    # [seed=42] mAP=0.7773 ... oof=8694s  train=88s  infer=4127s
                    metrics_match = re.search(r"\[seed=(\d+)\].*oof=(\d+)s\s+train=(\d+)s\s+infer=(\d+)s", line)
                    if metrics_match:
                        seed = int(metrics_match.group(1))
                        oof_t = float(metrics_match.group(2))
                        train_t = float(metrics_match.group(3))
                        infer_t = float(metrics_match.group(4))
                        timings[(current_strategy, seed)] = {
                            "oof_time_s": oof_t,
                            "train_time_s": train_t,
                            "infer_time_s": infer_t
                        }
    return timings

def collate_results():
    print("Loading ground truth data...")
    X_train, Y_train, X_test, Y_test = load_train_test()
    del X_train, X_test
    gc.collect()

    timings = parse_logs()

    all_results = {}
    all_probs = {}

    for strat_name in STRATEGIES:
        print(f"\nProcessing {strat_name}...")
        seed_metrics = {}
        probs_per_seed = []

        for seed in SEEDS:
            npy_path = PREDS_DIR / strat_name.lower() / f"{strat_name.lower()}_seed{seed}.npy"
            if not npy_path.exists():
                print(f"  [WARNING] Missing {npy_path}")
                continue
            
            data = np.load(str(npy_path), allow_pickle=True).item()
            train_oof_probs = data["train_oof_probs"]
            test_probs = data["test_probs"]

            metrics = compute_all_metrics(
                y_true=Y_test,
                y_probs=test_probs,
                label_names=LABEL_COLS,
                y_true_train=Y_train,
                y_probs_train=train_oof_probs,
            )

            t = timings.get((strat_name, seed), {"oof_time_s": 0.0, "train_time_s": 0.0, "infer_time_s": 0.0})
            metrics["oof_time_s"] = t["oof_time_s"]
            metrics["train_time_s"] = t["train_time_s"]
            metrics["infer_time_s"] = t["infer_time_s"]

            honest_f1 = metrics.get("macro_honest_f1", float("nan"))
            gap = metrics["macro_max_f1"] - honest_f1
            print(f"  [seed={seed}] mAP={metrics['macro_ap']:.4f}  honest_F1={honest_f1:.4f}  max_F1={metrics['macro_max_f1']:.4f}  gap={gap:.4f}")

            seed_metrics[seed] = metrics
            probs_per_seed.append((Y_test, test_probs))
        
        all_results[strat_name] = seed_metrics
        all_probs[strat_name] = probs_per_seed

    # ── save raw results ──────────────────────────────────────────────────────
    try:
        from src.utils.seed_utils import get_rng_states
        rng_states = get_rng_states()
    except:
        rng_states = {}
    
    tabpfn_version = "unknown"
    try:
        tabpfn_version = __import__("tabpfn").__version__
    except:
        pass

    raw_payload = {
        "run_metadata": {
            "seeds": SEEDS,
            "base_seed": SEEDS[0],
            "rng_states": rng_states,
            "tabpfn_version": tabpfn_version,
            "python_version": platform.python_version(),
            "note": "Collated from saved probabilities in data/preds_prob/"
        },
        "results": all_results,
    }

    raw_path = RESULTS_DIR / "benchmark_all_seeds.json"
    if raw_path.exists():
        raw_path.rename(raw_path.with_suffix(".json.bak"))
    with open(raw_path, "w") as f:
        json.dump(raw_payload, f, indent=2, default=float)
    print(f"\nSaved raw results → {raw_path}")

    # ── aggregate and build summary table ─────────────────────────────────────
    summary_rows = []
    br_aps = [all_results["BR"][s]["macro_ap"] for s in SEEDS] if "BR" in all_results and all_results["BR"] else []

    for strat_name in STRATEGIES:
        if strat_name not in all_results or not all_results[strat_name]:
            continue
        s_res = all_results[strat_name]

        macro_aps  = [s_res[s]["macro_ap"]                          for s in s_res]
        honest_f1s = [s_res[s].get("macro_honest_f1", float("nan")) for s in s_res]
        max_f1s    = [s_res[s]["macro_max_f1"]                      for s in s_res]
        sub_accs   = [s_res[s]["subset_accuracy"]                   for s in s_res]
        eces       = [s_res[s]["calibration"]["macro_ece"]          for s in s_res]
        briers     = [s_res[s]["calibration"]["macro_brier"]        for s in s_res]
        train_t    = [s_res[s]["train_time_s"]                      for s in s_res]
        infer_t    = [s_res[s]["infer_time_s"]                      for s in s_res]
        gaps       = [mf - hf for mf, hf in zip(max_f1s, honest_f1s) if not np.isnan(hf)]

        if all_probs.get(strat_name):
            Y_pool = np.vstack([p[0] for p in all_probs[strat_name]])
            P_pool = np.vstack([p[1] for p in all_probs[strat_name]])
            _, ci_lo, ci_hi = compute_bootstrap_ci(
                Y_pool, P_pool, metric_fn=compute_ap, n_bootstraps=2000
            )
            del Y_pool, P_pool
            gc.collect()
        else:
            ci_lo, ci_hi = float("nan"), float("nan")

        if br_aps and strat_name != "BR" and len(macro_aps) == len(br_aps):
            paired = compute_paired_statistical_tests(br_aps, macro_aps)
            delta_map = f"{paired['delta_mean']:+.4f}"
            wilcoxon_p = f"{paired['wilcoxon_p']:.3f}" if not np.isnan(paired.get("wilcoxon_p", float("nan"))) else "N/A"
        else:
            delta_map = "N/A" if strat_name != "BR" else "0.0000"
            wilcoxon_p = "N/A"

        summary_rows.append({
            "Strategy":            strat_name,
            "mAP (mean±std)":      f"{np.mean(macro_aps):.4f} ± {np.std(macro_aps):.4f}",
            "mAP 95% CI":          f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            "Honest-F1 (mean)":    f"{np.nanmean(honest_f1s):.4f}",
            "Max-F1 (mean)":       f"{np.mean(max_f1s):.4f}",
            "Gap (Max-Honest F1)": f"{np.mean(gaps):.4f}" if gaps else "N/A",
            "Subset Acc (mean)":   f"{np.mean(sub_accs):.4f}",
            "ECE (mean)":          f"{np.mean(eces):.4f}",
            "Brier (mean)":        f"{np.mean(briers):.4f}",
            "Δ mAP vs BR":         delta_map,
            "Wilcoxon p":          wilcoxon_p,
            "Train time (mean)":   f"{np.mean(train_t):.0f}s",
            "Infer time (mean)":   f"{np.mean(infer_t):.0f}s",
        })

    if not summary_rows:
        raise SystemExit(
            f"No cached predictions found under {PREDS_DIR}, so there is nothing to "
            f"collate.\nRun `python scripts/evaluate.py` first: it fits all five transforms for all three seeds and writes the cache. See docs/DATA.md section 3."
        )

    df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    if csv_path.exists():
        csv_path.rename(csv_path.with_suffix(".csv.bak"))
    df.to_csv(csv_path, index=False)

    print("\n" + "="*80)
    print("  TABPFN MULTILABEL BENCHMARK — Collated Summary")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nSaved summary → {csv_path}")

if __name__ == "__main__":
    collate_results()
