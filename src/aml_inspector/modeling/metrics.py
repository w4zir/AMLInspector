"""AML-oriented classification metrics and plots."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aml_inspector.modeling.config import DEFAULT_C_FALSE_ALARM, DEFAULT_C_MISS


def cost_weighted_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    c_miss: float = DEFAULT_C_MISS,
    c_false_alarm: float = DEFAULT_C_FALSE_ALARM,
) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    fn = np.sum(y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    return float(c_miss * fn + c_false_alarm * fp)


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_score >= threshold
    metrics: dict[str, float] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    metrics["tp"] = float(tp)
    return metrics


def precision_at_fraction(y_true: np.ndarray, y_score: np.ndarray, top_fraction: float) -> float:
    """Precision among top ``top_fraction`` highest scores."""
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(n * top_fraction)))
    order = np.argsort(-y_score)
    top = order[:k]
    return float(np.mean(y_true[top].astype(bool)))


def recall_at_budget(y_true: np.ndarray, y_score: np.ndarray, budget_fraction: float) -> float:
    """Recall when alerting top ``budget_fraction`` by score."""
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    k = max(1, int(np.ceil(len(y_true) * budget_fraction)))
    order = np.argsort(-y_score)
    alerted = np.zeros(len(y_true), dtype=bool)
    alerted[order[:k]] = True
    return float(np.sum(y_true & alerted) / n_pos)


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(bool), y_score)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bootstrap: int = 200,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap confidence interval for PR-AUC (useful on small holdout)."""
    rng = np.random.default_rng(random_state)
    y_true = y_true.astype(bool)
    if len(np.unique(y_true)) < 2:
        return {"pr_auc": float("nan"), "pr_auc_lo": float("nan"), "pr_auc_hi": float("nan")}
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(float(average_precision_score(yt, ys)))
    if not scores:
        base = float(average_precision_score(y_true, y_score))
        return {"pr_auc": base, "pr_auc_lo": base, "pr_auc_hi": base}
    arr = np.array(scores)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "pr_auc_lo": float(np.quantile(arr, alpha / 2)),
        "pr_auc_hi": float(np.quantile(arr, 1 - alpha / 2)),
    }


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}
