"""Tests for experimentation data prep, splits, imbalance, thresholds, and policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_inspector.data.datasets import (
    ENTITY_ID_COL,
    EVENT_TIMESTAMP_COL,
    LABEL_COL,
    TRANSACTION_ID_COL,
)
from aml_inspector.data.preprocess_home_bank import DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL
from aml_inspector.modeling.config import ExperimentBundle, ExperimentConfig, FrozenPolicy
from aml_inspector.modeling.data import NON_FEATURE_COLUMNS, load_manifest, manifest_feature_id
from aml_inspector.modeling.features import scale_pos_weight, select_feature_columns
from aml_inspector.modeling.metrics import binary_metrics, cost_weighted_error
from aml_inspector.modeling.policy import challenger_only_metrics, combined_policy_metrics
from aml_inspector.modeling.splits import stratified_train_val_split
from aml_inspector.modeling.thresholds import select_champion_threshold, select_challenger_threshold


def _synthetic_experiment_frame(n: int, *, split: str, pos_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    labels = rng.random(n) < pos_rate
    return pd.DataFrame(
        {
            TRANSACTION_ID_COL: [f"t{i}" for i in range(n)],
            EVENT_TIMESTAMP_COL: pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"),
            ENTITY_ID_COL: [f"a{i % 10}" for i in range(n)],
            "data_split_source": split,
            LABEL_COL: labels,
            "amount_usd": rng.uniform(1, 1000, size=n),
            "velocity_24h_outbound": rng.integers(0, 5, size=n),
            "is_outbound": rng.choice([True, False], size=n),
            "corridor_risk_score": rng.uniform(0, 1, size=n),
            "hrj_country_flag": rng.choice([True, False], size=n),
        }
    )


def test_select_feature_columns_excludes_ids_and_label() -> None:
    df = _synthetic_experiment_frame(20, split=DATA_SPLIT_MEDIUM)
    cols = select_feature_columns(df)
    assert LABEL_COL not in cols
    assert TRANSACTION_ID_COL not in cols
    assert ENTITY_ID_COL not in cols
    assert "amount_usd" in cols
    assert "velocity_24h_outbound" in cols


def test_non_feature_columns_cover_ids() -> None:
    assert TRANSACTION_ID_COL in NON_FEATURE_COLUMNS
    assert LABEL_COL in NON_FEATURE_COLUMNS


def test_stratified_split_preserves_classes_and_no_small() -> None:
    df = _synthetic_experiment_frame(500, split=DATA_SPLIT_MEDIUM, pos_rate=0.05)
    train, val = stratified_train_val_split(df, val_size=0.2, random_state=7)
    assert (train["data_split_source"] == DATA_SPLIT_MEDIUM).all()
    assert (val["data_split_source"] == DATA_SPLIT_MEDIUM).all()
    assert train[LABEL_COL].sum() >= 1
    assert val[LABEL_COL].sum() >= 1
    assert len(train) + len(val) == len(df)


def test_scale_pos_weight_imbalance() -> None:
    y = np.array([0, 0, 0, 0, 1], dtype=bool)
    assert scale_pos_weight(y) == pytest.approx(4.0)


def test_scale_pos_weight_zero_positives_raises() -> None:
    with pytest.raises(ValueError, match="zero positive"):
        scale_pos_weight(np.zeros(5, dtype=bool))


def test_select_champion_threshold_minimizes_cost() -> None:
    y = np.array([1, 1, 0, 0, 0], dtype=bool)
    prob = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    t, info = select_champion_threshold(y, prob, c_miss=10, c_false_alarm=1)
    assert 0.0 <= t <= 1.0
    assert "val_cost_weighted" in info


def test_select_challenger_threshold_budget() -> None:
    scores = np.arange(100, dtype=float)
    t = select_challenger_threshold(scores, budget_fraction=0.1)
    assert t == pytest.approx(90.0, rel=0.01)


def test_combined_policy_and_challenger_only_metrics() -> None:
    y = np.array([1, 0, 0, 1, 0], dtype=bool)
    champ = np.array([0.2, 0.8, 0.1, 0.3, 0.05])
    chall = np.array([0.5, 0.9, 0.1, 0.95, 0.2])
    policy = FrozenPolicy(t_champ=0.5, t_chall=0.85, t_champ_soft=0.4, policy_version="v1")
    co = challenger_only_metrics(y, champ, chall, policy)
    assert "challenger_only_flag_rate" in co
    assert "challenger_only_precision" in co
    combined = combined_policy_metrics(y, champ, chall, policy)
    assert "precision" in combined
    assert "challenger_only_flag_rate" in combined


def test_binary_metrics_with_single_class() -> None:
    y = np.zeros(10, dtype=bool)
    s = np.linspace(0, 1, 10)
    m = binary_metrics(y, s, threshold=0.5)
    assert m["precision"] == 0.0


def test_cost_weighted_error() -> None:
    y = np.array([1, 0, 0], dtype=bool)
    pred = np.array([0, 1, 0], dtype=bool)
    assert cost_weighted_error(y, pred, c_miss=10, c_false_alarm=1) == 11.0


def test_experiment_bundle_roundtrip_dict() -> None:
    from aml_inspector.modeling.artifacts import _bundle_from_dict
    from aml_inspector.modeling.config import SplitCounts

    bundle = ExperimentBundle(
        feature_columns=["amount_usd"],
        frozen_policy=FrozenPolicy(0.5, 0.8, 0.4, "v1"),
        experiment_config=ExperimentConfig(random_state=1),
        scale_pos_weight=4.0,
        medium_bank_id=70,
        small_bank_id=42,
        feature_manifest_id="abc_def",
        git_sha="sha",
        split_counts={"train": SplitCounts(100, 2, 0.02)},
    )
    restored = _bundle_from_dict(bundle.to_dict())
    assert restored.feature_columns == ["amount_usd"]
    assert restored.medium_bank_id == 70
    assert restored.frozen_policy.t_champ == 0.5


def test_manifest_feature_id(tmp_path: Path) -> None:
    m = {"git_sha": "abcdef123456", "feature_config_signature": "sig123"}
    assert manifest_feature_id(m) == "abcdef12_sig123"


def test_small_split_tag_in_synthetic_frame() -> None:
    df = _synthetic_experiment_frame(10, split=DATA_SPLIT_SMALL)
    assert (df["data_split_source"] == DATA_SPLIT_SMALL).all()


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")
