"""Tests for experimentation data prep, splits, imbalance, thresholds, and policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_inspector.data.datasets import (
    DATASET_TOKEN_HI_MEDIUM,
    DATASET_TOKEN_HI_SMALL,
    ENTITY_ID_COL,
    EVENT_TIMESTAMP_COL,
    FEATURE_BASE_ACCOUNT_DAILY,
    FEATURE_BASE_EXPERIMENT,
    LABEL_COL,
    TRANSACTION_ID_COL,
    bank_scoped_output_subdir,
    feature_output_processed_dir,
    feature_parquet_filename,
)
from aml_inspector.data.preprocess_home_bank import DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL
from aml_inspector.features.feature_build_config import load_feature_build_config
from aml_inspector.modeling.config import ExperimentBundle, ExperimentConfig, FrozenPolicy
from aml_inspector.modeling.data import (
    DATA_SPLIT_COL,
    NON_FEATURE_COLUMNS,
    SOURCE_BANK_COL,
    join_account_daily_features,
    load_all_bank_dataset_frames,
    load_combined_bank_frames,
    load_manifest,
    load_manifest_combined_frame,
    load_testing_bank_frames,
    load_training_bank_frames,
    manifest_feature_id,
)
from aml_inspector.modeling.features import (
    model_columns_for_flags,
    scale_pos_weight,
    select_feature_columns,
    select_feature_columns_from_config,
)
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


def _full_experiment_row(*, split: str, bank_id: int, i: int) -> dict:
    ts = pd.Timestamp("2022-01-01T00:00:00Z") + pd.Timedelta(hours=i)
    return {
        TRANSACTION_ID_COL: f"t{bank_id}_{i}",
        EVENT_TIMESTAMP_COL: ts,
        ENTITY_ID_COL: f"a{bank_id}",
        "data_split_source": split,
        LABEL_COL: i % 5 == 0,
        "is_outbound": True,
        "counterparty_bank_id": 200,
        "amount_usd": 100.0 + i,
        "amount_round_100": False,
        "corridor_risk_score": 0.5,
        "velocity_24h_outbound": 1,
        "dwell_sec_since_last_inbound": 3600.0,
        "fanout_unique_internal_7d": 2,
        "fanout_unique_internal_30d": 3,
        "graph_internal_degree": 1,
        "graph_component_size": 2,
        "company_age_days_proxy": 10.0,
        "pep_proxy_synthetic": False,
        "hrj_country_flag": False,
    }


def _write_bank_split_parquets(
    tmp_path: Path,
    *,
    bank_id: int,
    split: str,
    n_rows: int = 5,
    include_daily: bool = True,
) -> None:
    token = DATASET_TOKEN_HI_MEDIUM if split == DATA_SPLIT_MEDIUM else DATASET_TOKEN_HI_SMALL
    proc = feature_output_processed_dir(
        output_subdir=bank_scoped_output_subdir(bank_id=bank_id),
        processed_root=tmp_path / "processed",
    )
    proc.mkdir(parents=True, exist_ok=True)
    exp = pd.DataFrame([_full_experiment_row(split=split, bank_id=bank_id, i=i) for i in range(n_rows)])
    exp_name = feature_parquet_filename(
        bank_id=bank_id,
        dataset_token=token,
        base_name=FEATURE_BASE_EXPERIMENT,
    )
    exp.to_parquet(proc / exp_name, index=False)
    if include_daily:
        day = pd.Timestamp("2022-01-01T00:00:00Z")
        daily = pd.DataFrame(
            {
                ENTITY_ID_COL: [f"a{bank_id}"],
                EVENT_TIMESTAMP_COL: [day],
                "data_split_source": [split],
                "daily_tx_count": [n_rows],
                "daily_amount_sum": [float(n_rows * 100)],
            }
        )
        daily_name = feature_parquet_filename(
            bank_id=bank_id,
            dataset_token=token,
            base_name=FEATURE_BASE_ACCOUNT_DAILY,
        )
        daily.to_parquet(proc / daily_name, index=False)


def test_select_feature_columns_from_config_includes_account_daily(
    tmp_path: Path,
) -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "feature_build.json"
    loaded = load_feature_build_config(cfg_path)
    df = _synthetic_experiment_frame(20, split=DATA_SPLIT_MEDIUM)
    df["daily_tx_count"] = 1
    df["daily_amount_sum"] = 50.0
    for col in model_columns_for_flags(loaded.flags):
        if col not in df.columns and col not in ("daily_tx_count", "daily_amount_sum"):
            if col in ("is_outbound", "amount_round_100", "pep_proxy_synthetic", "hrj_country_flag"):
                df[col] = False
            elif col == "counterparty_bank_id":
                df[col] = 1
            else:
                df[col] = 0.0
    cols = select_feature_columns_from_config(df, loaded)
    assert "daily_tx_count" in cols
    assert "daily_amount_sum" in cols
    assert "amount_usd" in cols
    assert SOURCE_BANK_COL not in cols


def test_load_combined_bank_frames_skips_missing_bank(tmp_path: Path) -> None:
    _write_bank_split_parquets(tmp_path, bank_id=3, split=DATA_SPLIT_MEDIUM)
    _write_bank_split_parquets(tmp_path, bank_id=4, split=DATA_SPLIT_MEDIUM)
    result = load_combined_bank_frames(
        [3, 4, 99],
        split=DATA_SPLIT_MEDIUM,
        processed_root=tmp_path / "processed",
    )
    assert result.loaded_bank_ids == [3, 4]
    assert 99 in result.skipped_bank_ids
    assert len(result.frame) == 10
    assert set(result.frame[SOURCE_BANK_COL].unique()) == {3, 4}


def test_load_combined_bank_frames_skips_bank_without_daily(tmp_path: Path) -> None:
    _write_bank_split_parquets(tmp_path, bank_id=5, split=DATA_SPLIT_MEDIUM, include_daily=False)
    _write_bank_split_parquets(tmp_path, bank_id=6, split=DATA_SPLIT_MEDIUM)
    result = load_combined_bank_frames(
        [5, 6],
        split=DATA_SPLIT_MEDIUM,
        processed_root=tmp_path / "processed",
    )
    assert result.loaded_bank_ids == [6]
    assert 5 in result.skipped_bank_ids


def test_join_account_daily_features_adds_columns() -> None:
    exp = pd.DataFrame(
        {
            ENTITY_ID_COL: ["a1"],
            EVENT_TIMESTAMP_COL: [pd.Timestamp("2022-01-01T12:00:00Z")],
            "data_split_source": [DATA_SPLIT_MEDIUM],
        }
    )
    daily = pd.DataFrame(
        {
            ENTITY_ID_COL: ["a1"],
            EVENT_TIMESTAMP_COL: [pd.Timestamp("2022-01-01T00:00:00Z")],
            "data_split_source": [DATA_SPLIT_MEDIUM],
            "daily_tx_count": [3],
            "daily_amount_sum": [300.0],
        }
    )
    out = join_account_daily_features(exp, daily)
    assert out["daily_tx_count"].iloc[0] == 3
    assert out["daily_amount_sum"].iloc[0] == 300.0


def test_experiment_bundle_roundtrip_multi_bank_fields() -> None:
    from aml_inspector.modeling.artifacts import _bundle_from_dict
    from aml_inspector.modeling.config import SplitCounts

    bundle = ExperimentBundle(
        feature_columns=["amount_usd"],
        frozen_policy=FrozenPolicy(0.5, 0.8, 0.4, "v1"),
        experiment_config=ExperimentConfig(random_state=1),
        scale_pos_weight=4.0,
        medium_bank_id=3,
        small_bank_id=21,
        feature_manifest_id="multi_bank_abc",
        git_sha="sha",
        split_counts={"train": SplitCounts(100, 2, 0.02)},
        training_bank_ids=[3, 4, 5],
        loaded_training_bank_ids=[3, 4, 5],
        skipped_training_bank_ids=[6],
        feature_config_signature="sig123",
    )
    restored = _bundle_from_dict(bundle.to_dict())
    assert restored.training_bank_ids == [3, 4, 5]
    assert restored.skipped_training_bank_ids == [6]
    assert restored.feature_config_signature == "sig123"


def test_load_all_bank_dataset_frames_includes_both_splits(tmp_path: Path) -> None:
    _write_bank_split_parquets(tmp_path, bank_id=1, split=DATA_SPLIT_MEDIUM, n_rows=3)
    _write_bank_split_parquets(tmp_path, bank_id=1, split=DATA_SPLIT_SMALL, n_rows=4)
    result = load_all_bank_dataset_frames([1], processed_root=tmp_path / "processed")
    assert result.loaded_bank_ids == [1]
    assert len(result.frame) == 7
    assert set(result.frame[DATA_SPLIT_COL].unique()) == {DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL}
    assert (1, DATA_SPLIT_MEDIUM) in result.loaded_bank_splits
    assert (1, DATA_SPLIT_SMALL) in result.loaded_bank_splits


def test_load_all_bank_dataset_frames_partial_split_skip(tmp_path: Path) -> None:
    _write_bank_split_parquets(tmp_path, bank_id=2, split=DATA_SPLIT_MEDIUM, n_rows=3)
    result = load_all_bank_dataset_frames([2], processed_root=tmp_path / "processed")
    assert result.loaded_bank_ids == [2]
    assert 2 not in result.skipped_bank_ids
    assert len(result.frame) == 3
    assert (2, DATA_SPLIT_MEDIUM) in result.loaded_bank_splits
    assert (2, DATA_SPLIT_SMALL) in result.skipped_bank_splits


def test_load_all_bank_dataset_frames_multi_bank(tmp_path: Path) -> None:
    _write_bank_split_parquets(tmp_path, bank_id=10, split=DATA_SPLIT_MEDIUM)
    _write_bank_split_parquets(tmp_path, bank_id=10, split=DATA_SPLIT_SMALL)
    _write_bank_split_parquets(tmp_path, bank_id=11, split=DATA_SPLIT_MEDIUM)
    result = load_all_bank_dataset_frames([10, 11], processed_root=tmp_path / "processed")
    assert result.loaded_bank_ids == [10, 11]
    assert len(result.frame) == 15
    assert (11, DATA_SPLIT_SMALL) in result.skipped_bank_splits


def test_load_training_and_testing_bank_frames_are_combined(tmp_path: Path) -> None:
    _write_bank_split_parquets(tmp_path, bank_id=7, split=DATA_SPLIT_MEDIUM, n_rows=2)
    _write_bank_split_parquets(tmp_path, bank_id=7, split=DATA_SPLIT_SMALL, n_rows=2)
    train = load_training_bank_frames([7], processed_root=tmp_path / "processed")
    test = load_testing_bank_frames([7], processed_root=tmp_path / "processed")
    assert len(train.frame) == 4
    assert len(test.frame) == 4
    assert set(train.frame[DATA_SPLIT_COL].unique()) == {DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL}


def test_load_manifest_combined_frame_concatenates_both(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir(parents=True)
    med_path = proc / "4_HI_MEDIUM_experiment_entity_df.parquet"
    sml_path = proc / "4_HI_SMALL_experiment_entity_df.parquet"
    pd.DataFrame([_full_experiment_row(split=DATA_SPLIT_MEDIUM, bank_id=4, i=i) for i in range(3)]).to_parquet(
        med_path, index=False
    )
    pd.DataFrame([_full_experiment_row(split=DATA_SPLIT_SMALL, bank_id=4, i=i) for i in range(2)]).to_parquet(
        sml_path, index=False
    )
    manifest = {
        "medium_bank_id": 4,
        "small_bank_id": 4,
        "outputs": {
            "experiment_medium": str(med_path),
            "experiment_small": str(sml_path),
        },
    }
    combined, meta = load_manifest_combined_frame(manifest)
    assert meta["medium_bank_id"] == 4
    assert len(combined) == 5
    assert set(combined[DATA_SPLIT_COL].unique()) == {DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL}


def test_load_manifest_combined_frame_partial_outputs(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir(parents=True)
    med_path = proc / "5_HI_MEDIUM_experiment_entity_df.parquet"
    pd.DataFrame([_full_experiment_row(split=DATA_SPLIT_MEDIUM, bank_id=5, i=0)]).to_parquet(
        med_path, index=False
    )
    manifest = {
        "medium_bank_id": 5,
        "small_bank_id": 5,
        "outputs": {
            "experiment_medium": str(med_path),
            "experiment_small": str(proc / "missing_HI_SMALL_experiment_entity_df.parquet"),
        },
    }
    combined, _ = load_manifest_combined_frame(manifest)
    assert len(combined) == 1
    assert combined[DATA_SPLIT_COL].iloc[0] == DATA_SPLIT_MEDIUM
