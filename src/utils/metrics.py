"""
Evaluation metrics for multilabel classification.

Functions
---------
compute_ap                  — Average Precision per label and macro mAP
compute_max_f1              — Test-tuned maximum F1 per label
find_optimal_thresholds     — Best F1 threshold per label from OOF predictions
compute_honest_f1           — Threshold selected on train OOF, evaluated on test
compute_subset_accuracy     — Exact-match (subset) accuracy
compute_calibration_metrics — ECE, MCE, and Brier score per label
compute_all_metrics         — Full metrics suite in one call
compute_bootstrap_ci        — 95% percentile bootstrap CI on any metric function
compute_paired_statistical_tests — Paired Wilcoxon and t-test between two strategies
fast_ap                     — Low-overhead single-label AP (for bootstrap loops)
compute_bootstrap_ci_seed_averaged — Test-set bootstrap CI, seeds averaged within resample
compute_paired_bootstrap_delta     — Paired bootstrap on AP difference, identical resamples
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
)


def compute_ap(
    y_true: np.ndarray, y_probs: np.ndarray
) -> Tuple[List[float], float]:
    """
    Average Precision per label and macro-averaged AP (mAP).

    Returns
    -------
    (per_label_ap, macro_ap)
    """
    y_probs = np.nan_to_num(y_probs, nan=0.5)
    aps = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            aps.append(0.5)
        else:
            aps.append(float(average_precision_score(y_true[:, j], y_probs[:, j])))
    return aps, float(np.mean(aps))


def compute_max_f1(
    y_true: np.ndarray, y_probs: np.ndarray
) -> Tuple[List[float], float]:
    """
    Test-tuned maximum F1 per label (optimistic upper bound on threshold performance).

    Returns
    -------
    (per_label_max_f1, macro_max_f1)
    """
    y_probs = np.nan_to_num(y_probs, nan=0.5)
    max_f1s = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            max_f1s.append(0.0)
        else:
            precision, recall, _ = precision_recall_curve(y_true[:, j], y_probs[:, j])
            f1s = 2 * precision * recall / (precision + recall + 1e-8)
            max_f1s.append(float(np.max(f1s)))
    return max_f1s, float(np.mean(max_f1s))


def find_optimal_thresholds(
    y_true: np.ndarray, y_probs: np.ndarray
) -> List[float]:
    """
    Per-label optimal F1 threshold selected from OOF training predictions.
    Frozen before touching the test set to avoid data leakage.
    """
    y_probs = np.nan_to_num(y_probs, nan=0.5)
    thresholds = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            thresholds.append(0.5)
            continue
        precision, recall, thresh = precision_recall_curve(y_true[:, j], y_probs[:, j])
        if len(thresh) == 0:
            thresholds.append(0.5)
            continue
        f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        thresholds.append(float(thresh[np.argmax(f1s)]))
    return thresholds


def compute_honest_f1(
    y_true_train: np.ndarray,
    y_probs_train: np.ndarray,
    y_true_test: np.ndarray,
    y_probs_test: np.ndarray,
) -> Tuple[List[float], float, List[float]]:
    """
    Honest F1: thresholds selected on OOF train predictions, applied to test.

    Returns
    -------
    (per_label_honest_f1, macro_honest_f1, thresholds)
    """
    y_probs_train = np.nan_to_num(y_probs_train, nan=0.5)
    y_probs_test = np.nan_to_num(y_probs_test, nan=0.5)
    thresholds = find_optimal_thresholds(y_true_train, y_probs_train)
    honest_f1s = []
    for j in range(y_true_test.shape[1]):
        preds = (y_probs_test[:, j] >= thresholds[j]).astype(int)
        honest_f1s.append(float(f1_score(y_true_test[:, j], preds, zero_division=0)))
    return honest_f1s, float(np.mean(honest_f1s)), thresholds


def compute_subset_accuracy(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
    thresholds: Optional[List[float]] = None,
) -> float:
    """
    Exact-match multilabel subset accuracy.

    Args:
        threshold:  Single global cut applied to every label (default 0.5).
        thresholds: Optional per-label cuts. When given, overrides ``threshold``.
                    Pass the honest OOF thresholds here to avoid understating
                    performance on very low-prevalence labels, where a fixed
                    0.5 cut predicts the negative class almost everywhere.
    """
    if thresholds is not None:
        thr = np.asarray(thresholds, dtype=float).reshape(1, -1)
        y_pred = (y_probs >= thr).astype(int)
    else:
        y_pred = (y_probs >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


def compute_calibration_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15
) -> Dict[str, Union[List[float], float]]:
    """
    ECE, MCE, and Brier score per label and macro-averaged over B equal-width bins.

    Returns
    -------
    Dict with keys: ece_per_label, macro_ece, mce_per_label, macro_mce,
                    brier_per_label, macro_brier
    """
    n_labels = y_true.shape[1]
    bins = np.linspace(0, 1, n_bins + 1)
    ece_list, mce_list, brier_list = [], [], []

    for j in range(n_labels):
        y_t, y_p = y_true[:, j], y_probs[:, j]
        brier_list.append(float(brier_score_loss(y_t, y_p)))

        ece, max_gap = 0.0, 0.0
        n = len(y_t)
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            in_bin = (y_p >= lo) & (y_p <= hi if b == n_bins - 1 else y_p < hi)
            if in_bin.any():
                gap = abs(y_t[in_bin].mean() - y_p[in_bin].mean())
                ece += gap * in_bin.sum() / n
                max_gap = max(max_gap, gap)

        ece_list.append(float(ece))
        mce_list.append(float(max_gap))

    return {
        "ece_per_label": ece_list,
        "macro_ece": float(np.mean(ece_list)),
        "mce_per_label": mce_list,
        "macro_mce": float(np.mean(mce_list)),
        "brier_per_label": brier_list,
        "macro_brier": float(np.mean(brier_list)),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    label_names: Optional[List[str]] = None,
    y_true_train: Optional[np.ndarray] = None,
    y_probs_train: Optional[np.ndarray] = None,
    n_bins: int = 15,
) -> Dict:
    """
    Full evaluation suite: AP, Max-F1, Subset Accuracy, Calibration, and
    optionally Honest F1 (requires y_true_train and y_probs_train).
    """
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label_{j}" for j in range(n_labels)]

    aps, macro_ap = compute_ap(y_true, y_probs)
    max_f1s, macro_max_f1 = compute_max_f1(y_true, y_probs)
    subset_acc = compute_subset_accuracy(y_true, y_probs)
    calib = compute_calibration_metrics(y_true, y_probs, n_bins=n_bins)
    test_tuned_thresholds = find_optimal_thresholds(y_true, y_probs)

    res: Dict = {
        "macro_ap": macro_ap,
        "macro_max_f1": macro_max_f1,
        "subset_accuracy": subset_acc,
        "calibration": calib,
        "per_label": {
            label_names[j]: {
                "AP": aps[j], 
                "Max_F1": max_f1s[j],
                "Test_Tuned_Threshold": test_tuned_thresholds[j]
            }
            for j in range(min(n_labels, len(label_names)))
        },
    }

    if y_true_train is not None and y_probs_train is not None:
        honest_f1s, macro_honest_f1, thresholds = compute_honest_f1(
            y_true_train, y_probs_train, y_true, y_probs
        )
        res["macro_honest_f1"] = macro_honest_f1
        # Subset accuracy at the honest per-label cuts, alongside the 0.5 version.
        # The 0.5 cut is retained under the original key so existing result files
        # stay directly comparable.
        res["subset_accuracy_honest"] = compute_subset_accuracy(
            y_true, y_probs, thresholds=thresholds
        )
        for j in range(min(n_labels, len(honest_f1s))):
            res["per_label"][label_names[j]]["Honest_F1"] = honest_f1s[j]
            res["per_label"][label_names[j]]["Honest_Threshold"] = thresholds[j]

    return res


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric_fn: Callable = compute_ap,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    95% percentile bootstrap CI on macro metric (from metric_fn).

    WARNING — do not pass predictions pooled across seeds (e.g. ``np.vstack`` of
    3 seeds' probability matrices) to this function. Doing so resamples 3N rows
    from a 3N pool and counts every test peptide three times as if independent,
    which shrinks the interval by roughly sqrt(3). Use
    ``compute_bootstrap_ci_seed_averaged`` instead: it resamples the N real test
    rows once per iteration and averages the macro metric across seeds inside
    each resample.

    Returns
    -------
    (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        _, macro = metric_fn(y_true[idx], y_probs[idx])
        scores.append(macro)

    scores = np.sort(scores)
    alpha = (1.0 - confidence_level) / 2.0
    lo = int(np.floor(alpha * n_bootstraps))
    hi = int(np.ceil((1.0 - alpha) * n_bootstraps)) - 1
    return float(np.mean(scores)), float(scores[lo]), float(scores[hi])


def fast_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Average precision for one label, equivalent to sklearn's
    ``average_precision_score`` but without its per-call overhead.

    AP = (1 / n_pos) * Σ_k P@k · rel(k), the step-wise definition sklearn uses.
    Returns 0.5 for a degenerate (single-class) column, matching ``compute_ap``.
    Ties are broken by sort order rather than grouped; on continuous TabPFN
    probabilities this is numerically indistinguishable from sklearn (verified
    in tests/test_metrics.py).
    """
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return 0.5
    order = np.argsort(-np.asarray(y_score), kind="stable")
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / n_pos)


def _macro_ap_over_seeds(
    y_true: np.ndarray, probs_list: List[np.ndarray], idx: np.ndarray
) -> float:
    """Mean over seeds of the macro-AP computed on one bootstrap resample."""
    yt = y_true[idx]
    per_seed = [
        np.mean([fast_ap(yt[:, j], p[idx, j]) for j in range(yt.shape[1])])
        for p in probs_list
    ]
    return float(np.mean(per_seed))


def compute_bootstrap_ci_seed_averaged(
    y_true: np.ndarray,
    probs_list: Union[np.ndarray, List[np.ndarray]],
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Percentile bootstrap CI on macro-AP over the *real* test set.

    One resample of the N test rows per iteration; the macro-AP is computed for
    each seed's predictions on those same rows and averaged. Seeds are treated
    as repeated measurements of one test set, not as extra independent data —
    so the interval reflects test-set sampling uncertainty at the true n.

    Args:
        y_true:     (N, L) ground-truth label matrix.
        probs_list: One (N, L) probability matrix, or a list of them (one per seed).

    Returns
    -------
    Dict with point (full-sample estimate), mean, ci_lower, ci_upper, n_test.
    """
    if isinstance(probs_list, np.ndarray):
        probs_list = [probs_list]
    y_true = np.asarray(y_true)
    n = len(y_true)

    all_idx = np.arange(n)
    point = _macro_ap_over_seeds(y_true, probs_list, all_idx)

    rng = np.random.RandomState(seed)
    scores = np.empty(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        scores[b] = _macro_ap_over_seeds(
            y_true, probs_list, rng.choice(n, size=n, replace=True)
        )

    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.percentile(scores, [100 * alpha, 100 * (1.0 - alpha)])
    return {
        "point": point,
        "mean": float(np.mean(scores)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_test": int(n),
    }


def compute_paired_bootstrap_delta(
    y_true: np.ndarray,
    probs_a: Union[np.ndarray, List[np.ndarray]],
    probs_b: Union[np.ndarray, List[np.ndarray]],
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    label_index: Optional[int] = None,
) -> Dict[str, float]:
    """
    Paired bootstrap on the AP difference (B - A) using identical resamples.

    Both strategies are scored on the *same* resampled rows each iteration, so
    the shared test-set sampling variance cancels and the interval reflects only
    the between-method difference. This is the test the summary tables should
    report: with 3 seeds a paired Wilcoxon cannot return p < 0.25 regardless of
    effect size, whereas this resamples 16k+ test peptides.

    Args:
        probs_a / probs_b: (N, L) matrix or list of them (one per seed).
        label_index: If given, the delta is computed for that single label
                     instead of the macro average.

    Returns
    -------
    Dict with delta (full-sample), ci_lower, ci_upper, pseudo_p (fraction of
    resamples where the delta is <= 0), significant (CI excludes 0), n_test.
    """
    if isinstance(probs_a, np.ndarray):
        probs_a = [probs_a]
    if isinstance(probs_b, np.ndarray):
        probs_b = [probs_b]
    y_true = np.asarray(y_true)
    n = len(y_true)

    if label_index is None:
        def score(probs, idx):
            return _macro_ap_over_seeds(y_true, probs, idx)
    else:
        j = label_index

        def score(probs, idx):
            yt = y_true[idx, j]
            return float(np.mean([fast_ap(yt, p[idx, j]) for p in probs]))

    all_idx = np.arange(n)
    delta_point = score(probs_b, all_idx) - score(probs_a, all_idx)

    rng = np.random.RandomState(seed)
    deltas = np.empty(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)   # identical rows for A and B
        deltas[b] = score(probs_b, idx) - score(probs_a, idx)

    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.percentile(deltas, [100 * alpha, 100 * (1.0 - alpha)])
    return {
        "delta": float(delta_point),
        "delta_boot_mean": float(np.mean(deltas)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "pseudo_p": float(np.mean(deltas <= 0.0)),
        "significant": bool(lo > 0.0 or hi < 0.0),
        "n_test": int(n),
    }


def bootstrap_strategy_comparison(
    y_true: np.ndarray,
    strategy_probs: Dict[str, List[np.ndarray]],
    reference: str,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    label_names: Optional[List[str]] = None,
    macro_subsets: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    One shared-resample bootstrap pass covering every strategy at once.

    Each iteration draws a single resample of the N test rows and scores *all*
    strategies and labels on exactly those rows. Every reported CI and every
    pairwise delta therefore comes from identical resamples, so differences
    between strategies are properly paired and the shared test-set sampling
    variance cancels in the deltas.

    Seeds are averaged inside each resample — they are repeated measurements of
    one test set, not additional independent peptides.

    Args:
        y_true:         (N, L) ground-truth labels.
        strategy_probs: {strategy_name: [ (N, L) probs, one per seed ]}.
        reference:      Strategy name that deltas are computed against (e.g. "BR").
        label_names:    Names for the L label columns.
        macro_subsets:  Optional {name: [label subset]} macro averages to report.
                        A macro over a label subset is just the mean of the same
                        per-label APs, so any number of subsets (e.g. all five
                        labels vs. the four real activities with the deterministic
                        Antimicrobial OR-parent excluded) come out of this single
                        bootstrap pass at no extra cost. Defaults to one macro
                        over every label.

    Returns
    -------
    Dict with 'per_strategy' (per-subset macro and per-label point estimates with
    CIs) and 'deltas_vs_reference' (the same, as differences vs the reference).
    """
    y_true = np.asarray(y_true)
    n, n_labels = y_true.shape
    if label_names is None:
        label_names = [f"Label_{j}" for j in range(n_labels)]
    if reference not in strategy_probs:
        raise ValueError(f"Reference strategy {reference!r} not in strategy_probs")

    names = list(strategy_probs.keys())
    ref_i = names.index(reference)

    def _ap_grid(idx: np.ndarray) -> np.ndarray:
        """(n_strategies, n_labels) seed-averaged AP on one set of row indices."""
        yt = y_true[idx]
        out = np.empty((len(names), n_labels), dtype=float)
        for si, name in enumerate(names):
            seed_probs = [p[idx] for p in strategy_probs[name]]
            for j in range(n_labels):
                ytj = yt[:, j]
                out[si, j] = np.mean([fast_ap(ytj, p[:, j]) for p in seed_probs])
        return out

    point = _ap_grid(np.arange(n))

    rng = np.random.RandomState(seed)
    boot = np.empty((n_bootstraps, len(names), n_labels), dtype=float)
    for b in range(n_bootstraps):
        boot[b] = _ap_grid(rng.choice(n, size=n, replace=True))

    alpha = (1.0 - confidence_level) / 2.0
    pct = [100 * alpha, 100 * (1.0 - alpha)]

    if macro_subsets is None:
        macro_subsets = {"macro": list(label_names)}
    subset_cols = {}
    for sub_name, sub_labels in macro_subsets.items():
        missing = [l for l in sub_labels if l not in label_names]
        if missing:
            raise ValueError(f"macro_subsets[{sub_name!r}] has unknown labels: {missing}")
        subset_cols[sub_name] = [label_names.index(l) for l in sub_labels]

    per_strategy, deltas = {}, {}
    for si, name in enumerate(names):
        entry = {"macro": {}, "per_label": {}}
        for sub_name, cols in subset_cols.items():
            bm = boot[:, si, cols].mean(axis=1)
            lo, hi = np.percentile(bm, pct)
            entry["macro"][sub_name] = {
                "AP": float(point[si, cols].mean()),
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "n_labels": len(cols),
            }
        for j, lbl in enumerate(label_names):
            l_lo, l_hi = np.percentile(boot[:, si, j], pct)
            entry["per_label"][lbl] = {
                "AP": float(point[si, j]),
                "ci_lower": float(l_lo),
                "ci_upper": float(l_hi),
            }
        per_strategy[name] = entry

        if si == ref_i:
            continue

        d_entry = {"macro": {}, "per_label": {}}
        for sub_name, cols in subset_cols.items():
            dm = boot[:, si, cols].mean(axis=1) - boot[:, ref_i, cols].mean(axis=1)
            d_lo, d_hi = np.percentile(dm, pct)
            d_entry["macro"][sub_name] = {
                "delta_AP": float(point[si, cols].mean() - point[ref_i, cols].mean()),
                "ci_lower": float(d_lo),
                "ci_upper": float(d_hi),
                "pseudo_p": float(np.mean(dm <= 0.0)),
                "significant": bool(d_lo > 0.0 or d_hi < 0.0),
            }
        for j, lbl in enumerate(label_names):
            dj = boot[:, si, j] - boot[:, ref_i, j]
            j_lo, j_hi = np.percentile(dj, pct)
            d_entry["per_label"][lbl] = {
                "delta_AP": float(point[si, j] - point[ref_i, j]),
                "ci_lower": float(j_lo),
                "ci_upper": float(j_hi),
                "pseudo_p": float(np.mean(dj <= 0.0)),
                "significant": bool(j_lo > 0.0 or j_hi < 0.0),
            }
        deltas[name] = d_entry

    return {
        "reference": reference,
        "n_test": int(n),
        "n_bootstraps": int(n_bootstraps),
        "confidence_level": confidence_level,
        "label_names": list(label_names),
        "macro_subsets": {k: list(v) for k, v in macro_subsets.items()},
        "per_strategy": per_strategy,
        "deltas_vs_reference": deltas,
    }


def compute_paired_statistical_tests(
    vals_a: Union[List[float], np.ndarray],
    vals_b: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test and paired t-test between strategy A and B.

    Args:
        vals_a: Macro metric scores for strategy A across seeds.
        vals_b: Macro metric scores for strategy B across seeds.

    Returns
    -------
    Dict with delta_mean, delta_std, n_samples, wilcoxon_p, ttest_p.
    """
    a, b = np.asarray(vals_a, dtype=float), np.asarray(vals_b, dtype=float)
    diff = b - a
    out = {
        "delta_mean": float(np.mean(diff)),
        "delta_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
        "n_samples": len(diff),
    }

    try:
        if np.any(diff != 0):
            _, p = wilcoxon(b, a)
            out["wilcoxon_p"] = float(p)
        else:
            out["wilcoxon_p"] = float("nan")
    except Exception:
        out["wilcoxon_p"] = float("nan")

    try:
        if len(diff) > 1:
            _, p = ttest_rel(b, a)
            out["ttest_p"] = float(p)
        else:
            out["ttest_p"] = float("nan")
    except Exception:
        out["ttest_p"] = float("nan")

    return out
