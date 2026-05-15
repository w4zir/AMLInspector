"""Champion–Challenger combined alert policy."""

from __future__ import annotations

import numpy as np

from aml_inspector.modeling.config import FrozenPolicy
from aml_inspector.modeling.metrics import binary_metrics


def combined_alerts(
    champion_prob: np.ndarray,
    challenger_score: np.ndarray,
    policy: FrozenPolicy,
) -> np.ndarray:
    """Alert if champion >= T_champ OR challenger >= T_chall."""
    return (champion_prob >= policy.t_champ) | (challenger_score >= policy.t_chall)


def champion_miss_queue(
    champion_prob: np.ndarray,
    challenger_score: np.ndarray,
    policy: FrozenPolicy,
) -> np.ndarray:
    """High-priority review: challenger high while champion below soft threshold."""
    return (challenger_score >= policy.t_chall) & (champion_prob < policy.t_champ_soft)


def challenger_only_metrics(
    y_true: np.ndarray,
    champion_prob: np.ndarray,
    challenger_score: np.ndarray,
    policy: FrozenPolicy,
) -> dict[str, float]:
    y_true = y_true.astype(bool)
    champ_pos = champion_prob >= policy.t_champ
    chall_pos = challenger_score >= policy.t_chall
    challenger_only = chall_pos & ~champ_pos
    n = len(y_true)
    rate = float(np.mean(challenger_only)) if n else 0.0
    if challenger_only.any():
        precision = float(np.mean(y_true[challenger_only]))
    else:
        precision = 0.0
    return {
        "challenger_only_flag_rate": rate,
        "challenger_only_precision": precision,
        "challenger_only_count": float(np.sum(challenger_only)),
    }


def combined_policy_metrics(
    y_true: np.ndarray,
    champion_prob: np.ndarray,
    challenger_score: np.ndarray,
    policy: FrozenPolicy,
) -> dict[str, float]:
    alerts = combined_alerts(champion_prob, challenger_score, policy)
    out = binary_metrics(y_true, alerts.astype(float), threshold=0.5)
    out.update(challenger_only_metrics(y_true, champion_prob, challenger_score, policy))
    miss = champion_miss_queue(champion_prob, challenger_score, policy)
    out["champion_miss_queue_rate"] = float(np.mean(miss)) if len(miss) else 0.0
    if miss.any():
        out["champion_miss_queue_precision"] = float(np.mean(y_true[miss]))
    else:
        out["champion_miss_queue_precision"] = 0.0
    return out
