"""End-to-end test for preprocess + feature build (tiny synthetic CSVs)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aml_inspector.data.datasets import (
    DATASET_TOKEN_HI_MEDIUM,
    DATASET_TOKEN_HI_SMALL,
    FEATURE_BASE_ACCOUNT_DAILY,
    FEATURE_BASE_EXPERIMENT,
    FEATURE_BASE_TXN_LEVEL,
    MANIFEST_JSON,
    PARQUET_HOME_MEDIUM,
    PARQUET_HOME_SMALL,
    feature_parquet_filename,
    feature_parquet_paths,
)
from aml_inspector.data.preprocess_home_bank import run_preprocess_medium_small
from aml_inspector.features.build_tables import build_all_feature_tables, feature_artifacts_present
from aml_inspector.features.feature_build_config import (
    default_feature_build_config_path,
    load_feature_build_config,
)


def _ibm_header() -> str:
    return "Timestamp,From Bank,Account,To Bank,Account,Amount Paid,Payment Currency,Is Laundering\n"


def test_build_feature_tables_medium_small_split(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    interim = tmp_path / "interim"
    raw.mkdir()
    proc.mkdir()
    interim.mkdir()

    (raw / "HI-Medium_Trans.csv").write_text(
        _ibm_header()
        + "2022-01-01 00:00:00,100,1,200,2,500,USD,0\n"
        "2022-01-01 02:00:00,100,1,100,3,100,USD,0\n"  # internal
        "2022-01-01 04:00:00,200,2,100,1,50,USD,1\n",
        encoding="utf-8",
    )
    (raw / "HI-Small_Trans.csv").write_text(
        _ibm_header()
        + "2022-02-01 00:00:00,100,1,200,9,10,USD,0\n",
        encoding="utf-8",
    )

    summary = run_preprocess_medium_small(
        raw_dir=raw,
        bank_id=100,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        output_medium=proc / PARQUET_HOME_MEDIUM,
        output_small=proc / PARQUET_HOME_SMALL,
        summary_json=interim / "home_bank_selection_summary.json",
    )
    assert summary["home_bank_id"] == 100
    assert summary["medium_bank_id"] == 100
    assert summary["small_bank_id"] == 100

    manifest = build_all_feature_tables(
        medium_bank_id=100,
        small_bank_id=100,
        path_medium=proc / PARQUET_HOME_MEDIUM,
        path_small=proc / PARQUET_HOME_SMALL,
        interim_dir=interim,
        processed_dir=proc,
        preprocess_summary=summary,
    )
    assert (interim / MANIFEST_JSON).is_file()
    assert "outputs" in manifest

    paths = feature_parquet_paths(medium_bank_id=100, small_bank_id=100, processed_dir=proc)
    exp_m = pd.read_parquet(paths["experiment_medium"])
    exp_s = pd.read_parquet(paths["experiment_small"])
    assert (exp_m["data_split_source"] == "medium").all()
    assert (exp_s["data_split_source"] == "small").all()
    assert "transaction_id" in exp_m.columns
    assert "velocity_24h_outbound" in exp_m.columns

    expected_daily = feature_parquet_filename(
        bank_id=100,
        dataset_token=DATASET_TOKEN_HI_MEDIUM,
        base_name=FEATURE_BASE_ACCOUNT_DAILY,
    )
    assert paths["account_daily_medium"].name == expected_daily
    assert paths["account_daily_medium"].is_file()

    meta = json.loads((interim / MANIFEST_JSON).read_text(encoding="utf-8"))
    assert meta["medium_bank_id"] == 100
    assert meta["small_bank_id"] == 100
    assert meta["home_bank_id"] == 100
    assert "feature_config_signature" in meta
    assert meta["feature_config"]["bank_filter"]["medium"]["rows_after"] == 3
    assert meta["feature_config"]["bank_filter"]["small"]["rows_after"] == 1


def test_build_feature_tables_separate_medium_small_banks(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    interim = tmp_path / "interim"
    raw.mkdir()
    proc.mkdir()
    interim.mkdir()

    (raw / "HI-Medium_Trans.csv").write_text(
        _ibm_header()
        + "2022-01-01 00:00:00,70,1,200,2,500,USD,0\n"
        + "2022-01-01 02:00:00,70,1,70,3,100,USD,0\n",
        encoding="utf-8",
    )
    (raw / "HI-Small_Trans.csv").write_text(
        _ibm_header() + "2022-02-01 00:00:00,42,1,200,9,10,USD,0\n",
        encoding="utf-8",
    )

    summary = run_preprocess_medium_small(
        raw_dir=raw,
        medium_bank_id=70,
        small_bank_id=42,
        output_medium=proc / PARQUET_HOME_MEDIUM,
        output_small=proc / PARQUET_HOME_SMALL,
        summary_json=interim / "home_bank_selection_summary.json",
    )
    assert summary["medium_bank_id"] == 70
    assert summary["small_bank_id"] == 42
    assert "home_bank_id" not in summary

    build_all_feature_tables(
        medium_bank_id=70,
        small_bank_id=42,
        path_medium=proc / PARQUET_HOME_MEDIUM,
        path_small=proc / PARQUET_HOME_SMALL,
        interim_dir=interim,
        processed_dir=proc,
        preprocess_summary=summary,
    )

    paths = feature_parquet_paths(medium_bank_id=70, small_bank_id=42, processed_dir=proc)
    assert paths["txn_level_medium"].name == feature_parquet_filename(
        bank_id=70,
        dataset_token=DATASET_TOKEN_HI_MEDIUM,
        base_name=FEATURE_BASE_TXN_LEVEL,
    )
    assert paths["txn_level_small"].name == feature_parquet_filename(
        bank_id=42,
        dataset_token=DATASET_TOKEN_HI_SMALL,
        base_name=FEATURE_BASE_TXN_LEVEL,
    )
    assert paths["experiment_small"].name == feature_parquet_filename(
        bank_id=42,
        dataset_token=DATASET_TOKEN_HI_SMALL,
        base_name=FEATURE_BASE_EXPERIMENT,
    )
    txn_med = pd.read_parquet(paths["txn_level_medium"])
    txn_sml = pd.read_parquet(paths["txn_level_small"])
    assert len(txn_med) == 2
    assert len(txn_sml) == 1
    assert (txn_med["from_bank"] == 70).any() or (txn_med["to_bank"] == 70).any()
    assert (txn_sml["from_bank"] == 42).any() or (txn_sml["to_bank"] == 42).any()


def test_bank_filter_drops_rows_not_involving_home_bank(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    interim = tmp_path / "interim"
    raw.mkdir()
    proc.mkdir()
    interim.mkdir()
    (raw / "HI-Medium_Trans.csv").write_text(
        _ibm_header()
        + "2022-01-01 00:00:00,100,1,200,2,500,USD,0\n"
        "2022-01-01 02:00:00,100,1,100,3,100,USD,0\n"
        "2022-01-01 04:00:00,200,2,100,1,50,USD,1\n",
        encoding="utf-8",
    )
    (raw / "HI-Small_Trans.csv").write_text(
        _ibm_header() + "2022-02-01 00:00:00,100,1,200,9,10,USD,0\n",
        encoding="utf-8",
    )
    summary = run_preprocess_medium_small(
        raw_dir=raw,
        bank_id=100,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        output_medium=proc / PARQUET_HOME_MEDIUM,
        output_small=proc / PARQUET_HOME_SMALL,
        summary_json=interim / "home_bank_selection_summary.json",
    )
    med_df = pd.read_parquet(proc / PARQUET_HOME_MEDIUM)
    junk = med_df.iloc[[0]].copy()
    junk["Timestamp"] = "2030-01-01 00:00:00"
    junk["From Bank"] = 300
    junk["To Bank"] = 400
    if "event_timestamp" in junk.columns:
        junk["event_timestamp"] = pd.Timestamp("2030-01-01T00:00:00Z")
    pd.concat([med_df, junk], ignore_index=True).to_parquet(proc / PARQUET_HOME_MEDIUM, index=False)

    build_all_feature_tables(
        medium_bank_id=100,
        small_bank_id=100,
        path_medium=proc / PARQUET_HOME_MEDIUM,
        path_small=proc / PARQUET_HOME_SMALL,
        interim_dir=interim,
        processed_dir=proc,
        preprocess_summary=summary,
    )
    paths = feature_parquet_paths(medium_bank_id=100, small_bank_id=100, processed_dir=proc)
    txn = pd.read_parquet(paths["txn_level_medium"])
    assert len(txn) == 3
    assert not ((txn["from_bank"] == 300) & (txn["to_bank"] == 400)).any()


def test_disabled_rolling_and_graph_use_defaults(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    interim = tmp_path / "interim"
    raw.mkdir()
    proc.mkdir()
    interim.mkdir()
    (raw / "HI-Medium_Trans.csv").write_text(
        _ibm_header()
        + "2022-01-01 00:00:00,100,1,200,2,500,USD,0\n"
        "2022-01-01 02:00:00,100,1,100,3,100,USD,0\n",
        encoding="utf-8",
    )
    (raw / "HI-Small_Trans.csv").write_text(
        _ibm_header() + "2022-02-01 00:00:00,100,1,200,9,10,USD,0\n",
        encoding="utf-8",
    )
    summary = run_preprocess_medium_small(
        raw_dir=raw,
        bank_id=100,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        output_medium=proc / PARQUET_HOME_MEDIUM,
        output_small=proc / PARQUET_HOME_SMALL,
        summary_json=interim / "home_bank_selection_summary.json",
    )
    doc = json.loads(default_feature_build_config_path().read_text(encoding="utf-8"))
    doc["features"]["weekly_internal_graph"]["enabled"] = False
    doc["features"]["rolling_account_activity"]["enabled"] = False
    fc_path = tmp_path / "feature_build.json"
    fc_path.write_text(json.dumps(doc), encoding="utf-8")
    cfg = load_feature_build_config(fc_path)

    build_all_feature_tables(
        medium_bank_id=100,
        small_bank_id=100,
        path_medium=proc / PARQUET_HOME_MEDIUM,
        path_small=proc / PARQUET_HOME_SMALL,
        interim_dir=interim,
        processed_dir=proc,
        preprocess_summary=summary,
        feature_build_config=cfg,
    )
    paths = feature_parquet_paths(medium_bank_id=100, small_bank_id=100, processed_dir=proc)
    txn = pd.concat(
        [
            pd.read_parquet(paths["txn_level_medium"]),
            pd.read_parquet(paths["txn_level_small"]),
        ],
        ignore_index=True,
    )
    assert (txn["velocity_24h_outbound"] == 0).all()
    assert txn["dwell_sec_since_last_inbound"].isna().all()
    assert (txn["fanout_unique_internal_7d"] == 0).all()
    assert (txn["fanout_unique_internal_30d"] == 0).all()
    assert (txn["graph_internal_degree"] == 0).all()
    assert (txn["graph_component_size"] == 1).all()


def test_feature_artifacts_present_requires_all_parquets(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    interim = tmp_path / "interim"
    proc.mkdir()
    interim.mkdir()
    assert (
        feature_artifacts_present(
            medium_bank_id=1,
            small_bank_id=2,
            processed_dir=proc,
            interim_dir=interim,
        )
        is None
    )

    manifest = {"medium_bank_id": 1, "small_bank_id": 2, "outputs": {}}
    (interim / MANIFEST_JSON).write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        feature_artifacts_present(
            medium_bank_id=1,
            small_bank_id=2,
            processed_dir=proc,
            interim_dir=interim,
        )
        is None
    )

    for path in feature_parquet_paths(
        medium_bank_id=1, small_bank_id=2, processed_dir=proc
    ).values():
        path.write_bytes(b"x")
    loaded = feature_artifacts_present(
        medium_bank_id=1,
        small_bank_id=2,
        processed_dir=proc,
        interim_dir=interim,
    )
    assert loaded is not None
    assert loaded["medium_bank_id"] == 1
    assert loaded["small_bank_id"] == 2

    assert (
        feature_artifacts_present(
            medium_bank_id=9,
            small_bank_id=2,
            processed_dir=proc,
            interim_dir=interim,
        )
        is None
    )


def test_feature_artifacts_present_signature_mismatch(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    interim = tmp_path / "interim"
    proc.mkdir()
    interim.mkdir()
    manifest = {
        "medium_bank_id": 1,
        "small_bank_id": 2,
        "feature_config_signature": "aaaaaaaaaaaaaaaa",
        "outputs": {},
    }
    (interim / MANIFEST_JSON).write_text(json.dumps(manifest), encoding="utf-8")
    for path in feature_parquet_paths(
        medium_bank_id=1, small_bank_id=2, processed_dir=proc
    ).values():
        path.write_bytes(b"x")
    assert (
        feature_artifacts_present(
            medium_bank_id=1,
            small_bank_id=2,
            processed_dir=proc,
            interim_dir=interim,
            feature_config_signature="bbbbbbbbbbbbbbbb",
        )
        is None
    )


def test_feature_artifacts_present_signature_match(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    interim = tmp_path / "interim"
    proc.mkdir()
    interim.mkdir()
    sig = "cccccccccccccccc"
    manifest = {
        "medium_bank_id": 1,
        "small_bank_id": 2,
        "feature_config_signature": sig,
        "outputs": {},
    }
    (interim / MANIFEST_JSON).write_text(json.dumps(manifest), encoding="utf-8")
    for path in feature_parquet_paths(
        medium_bank_id=1, small_bank_id=2, processed_dir=proc
    ).values():
        path.write_bytes(b"x")
    loaded = feature_artifacts_present(
        medium_bank_id=1,
        small_bank_id=2,
        processed_dir=proc,
        interim_dir=interim,
        feature_config_signature=sig,
    )
    assert loaded is not None
    assert loaded["feature_config_signature"] == sig
