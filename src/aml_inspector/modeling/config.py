"""Experiment configuration constants and dataclass."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# AML cost model (validation-only threshold tuning)
DEFAULT_C_MISS = 10.0
DEFAULT_C_FALSE_ALARM = 1.0

DEFAULT_RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.2
DEFAULT_REVIEW_BUDGET_FRACTION = 0.01  # top 1% by challenger score if budget not set

MLFLOW_EXPERIMENT_CHAMPION = "aml_champion"
MLFLOW_EXPERIMENT_CHALLENGER = "aml_challenger"
MLFLOW_EXPERIMENT_COMBINED = "aml_champion_challenger"

SPLIT_POLICY_TAG = "medium_80_20_small_holdout"


@dataclass
class ExperimentConfig:
    """Frozen experiment settings logged with each MLflow run."""

    random_state: int = DEFAULT_RANDOM_STATE
    val_size: float = DEFAULT_VAL_SIZE
    c_miss: float = DEFAULT_C_MISS
    c_false_alarm: float = DEFAULT_C_FALSE_ALARM
    review_budget_fraction: float = DEFAULT_REVIEW_BUDGET_FRACTION
    split_policy: str = SPLIT_POLICY_TAG
    champion_soft_threshold_margin: float = 0.05
    policy_version: str = "champion_or_challenger_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrozenPolicy:
    """Operating thresholds chosen on validation only."""

    t_champ: float
    t_chall: float
    t_champ_soft: float
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitCounts:
    """Row and positive-rate summary per split."""

    n_rows: int
    n_positives: int
    positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentBundle:
    """Serializable artifact bundle for train → evaluate."""

    feature_columns: list[str]
    frozen_policy: FrozenPolicy
    experiment_config: ExperimentConfig
    scale_pos_weight: float
    medium_bank_id: int
    small_bank_id: int
    feature_manifest_id: str
    git_sha: str
    split_counts: dict[str, SplitCounts]
    champion_params: dict[str, Any] = field(default_factory=dict)
    challenger_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "frozen_policy": self.frozen_policy.to_dict(),
            "experiment_config": self.experiment_config.to_dict(),
            "scale_pos_weight": self.scale_pos_weight,
            "medium_bank_id": self.medium_bank_id,
            "small_bank_id": self.small_bank_id,
            "feature_manifest_id": self.feature_manifest_id,
            "git_sha": self.git_sha,
            "split_counts": {k: v.to_dict() for k, v in self.split_counts.items()},
            "champion_params": self.champion_params,
            "challenger_params": self.challenger_params,
        }
