"""Disease-level discrimination and operating-point metrics."""

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def discrimination_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    prevalence = float(labels.mean()) if labels.size else math.nan
    if labels.size == 0 or np.unique(labels).size < 2:
        return {
            "n": int(labels.size),
            "positive_n": int(labels.sum()),
            "prevalence": prevalence,
            "auroc": math.nan,
            "auprc": math.nan,
            "normalized_auprc": math.nan,
        }
    auprc = float(average_precision_score(labels, scores))
    return {
        "n": int(labels.size),
        "positive_n": int(labels.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": auprc,
        "normalized_auprc": auprc / prevalence if prevalence > 0 else math.nan,
    }


def operating_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(scores) >= float(threshold)
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & negative))
    fn = int(np.sum(~predictions & positive))
    tn = int(np.sum(~predictions & negative))

    def divide(numerator, denominator):
        return numerator / denominator if denominator else math.nan

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    ppv = divide(tp, tp + fp)
    npv = divide(tn, tn + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "lr_positive": divide(sensitivity, 1 - specificity),
        "lr_negative": divide(1 - sensitivity, specificity),
        "mcc": divide(tp * tn - fp * fn, denominator),
    }


def _threshold_table(labels, scores):
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    tp = np.cumsum(ordered_labels == 1)
    fp = np.cumsum(ordered_labels == 0)
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), labels.size - 1]
    tp = tp[ends].astype(np.float64)
    fp = fp[ends].astype(np.float64)
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    fn = positives - tp
    tn = negatives - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.full_like(tp, np.nan),
        where=denominator > 0,
    )
    sensitivity = tp / positives
    specificity = tn / negatives
    return ordered_scores[ends], mcc, sensitivity, specificity


def select_mcc_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return math.nan
    thresholds, mcc, _, _ = _threshold_table(labels, scores)
    finite = np.flatnonzero(np.isfinite(mcc))
    if finite.size == 0:
        return math.nan
    best = finite[np.argmax(mcc[finite])]
    return float(thresholds[best])


def select_sensitivity_threshold(labels, scores, target):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return math.nan
    thresholds, _, sensitivity, specificity = _threshold_table(labels, scores)
    eligible = np.flatnonzero(sensitivity >= target)
    if eligible.size == 0:
        return math.nan
    best = eligible[np.argmax(specificity[eligible])]
    return float(thresholds[best])


def bootstrap_intervals(labels, scores, repetitions=1000, seed=42):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if repetitions <= 0 or labels.size == 0 or np.unique(labels).size < 2:
        return {}
    rng = np.random.default_rng(seed)
    estimates = {"auroc": [], "auprc": [], "normalized_auprc": []}
    for _ in range(repetitions):
        indices = rng.integers(0, labels.size, labels.size)
        metrics = discrimination_metrics(labels[indices], scores[indices])
        for name, values in estimates.items():
            if np.isfinite(metrics[name]):
                values.append(metrics[name])
    intervals = {}
    for name, values in estimates.items():
        if values:
            lower, upper = np.quantile(values, [0.025, 0.975])
            intervals[f"{name}_ci_low"] = float(lower)
            intervals[f"{name}_ci_high"] = float(upper)
    return intervals
