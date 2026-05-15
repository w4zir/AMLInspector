"""Threshold selection on validation (never on holdout)."""

from __future__ import annotations

import numpy as np

from aml_inspector.modeling.config import DEFAULT_C_FALSE_ALARM, DEFAULT_C_MISS
from aml_inspector.modeling.metrics import cost_weighted_error


def select_champion_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    c_miss: float = DEFAULT_C_MISS,
    c_false_alarm: float = DEFAULT_C_FALSE_ALARM,
    n_thresholds: int = 101,
) -> tuple[float, dict[str, float]]:
    """Pick threshold minimizing cost-weighted error on validation."""
    y_true = y_true.astype(bool)
    if len(np.unique(y_prob)) == 1:
        t = float(y_prob[0])
        pred = y_prob >= t
        cost = cost_weighted_error(y_true, pred, c_miss=c_miss, c_false_alarm=c_false_alarm)
        return t, {"val_cost_weighted": cost}

    lo, hi = float(np.min(y_prob)), float(np.max(y_prob))
    candidates = np.linspace(lo, hi, n_thresholds)
    best_t = 0.5
    best_cost = float("inf")
    for t in candidates:
        pred = y_prob >= t
        cost = cost_weighted_error(y_true, pred, c_miss=c_miss, c_false_alarm=c_false_alarm)
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t, {"val_cost_weighted": best_cost}


def select_challenger_threshold(
    y_score: np.ndarray,
    *,
    budget_fraction: float,
) -> float:
    """Score threshold at the (1 - budget_fraction) quantile (top fraction alerts)."""
    frac = min(max(budget_fraction, 1e-6), 1.0)
    q = 1.0 - frac
    return float(np.quantile(y_score, q))


def champion_soft_threshold(t_champ: float, margin: float) -> float:
    return max(0.0, float(t_champ) - float(margin))
