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
    bank_scoped_output_subdir,
    feature_output_interim_dir,
    feature_output_processed_dir,
    feature_parquet_filename,
    feature_parquet_paths,
    resolve_feature_output_subdir,
    split_pair_scoped_output_subdir,
)
from aml_inspector.data.preprocess_home_bank import run_preprocess_medium_small
from aml_inspector.features.build_tables import build_all_feature_tables, feature_artifacts_present
from aml_inspector.features.feature_build_config import (
    default_feature_build_config_path,
    load_feature_build_config,
)
from aml_inspector.pipelines.build_feature_data import (
    FeatureBuildRunSpec,
    _bank_ids_from_spec,
    plan_feature_build_runs,
    run_feature_build,
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


def test_resolve_feature_output_subdir() -> None:
    assert resolve_feature_output_subdir(medium_bank_id=4, small_bank_id=4) == "bank_4"
    assert (
        resolve_feature_output_subdir(medium_bank_id=70, small_bank_id=42)
        == "medium_70_small_42"
    )


def test_plan_feature_build_runs_multi_bank() -> None:
    runs = plan_feature_build_runs(bank_ids=[4, 70], medium_bank_id=None, small_bank_id=None)
    assert len(runs) == 2
    assert runs[0].output_subdir == bank_scoped_output_subdir(bank_id=4)
    assert runs[0].bank_id == 4
    assert runs[1].output_subdir == bank_scoped_output_subdir(bank_id=70)


def test_plan_feature_build_runs_split_pair() -> None:
    runs = plan_feature_build_runs(bank_ids=None, medium_bank_id=70, small_bank_id=42)
    assert len(runs) == 1
    assert runs[0].output_subdir == split_pair_scoped_output_subdir(
        medium_bank_id=70, small_bank_id=42
    )


def test_plan_feature_build_runs_rejects_mixed_modes() -> None:
    import pytest

    with pytest.raises(ValueError, match="Cannot combine"):
        plan_feature_build_runs(bank_ids=[4], medium_bank_id=70, small_bank_id=42)


def test_bank_ids_from_spec() -> None:
    assert _bank_ids_from_spec(FeatureBuildRunSpec(output_subdir="bank_4", bank_id=4)) == (
        4,
        4,
        "bank_4",
    )
    assert _bank_ids_from_spec(
        FeatureBuildRunSpec(
            output_subdir="medium_70_small_42",
            medium_bank_id=70,
            small_bank_id=42,
        )
    ) == (70, 42, "medium_70_small_42")
    assert _bank_ids_from_spec(FeatureBuildRunSpec(output_subdir=None)) is None


def _write_tiny_raw_csvs(raw: Path, *, medium_bank: int, small_bank: int) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "HI-Medium_Trans.csv").write_text(
        _ibm_header()
        + f"2022-01-01 00:00:00,{medium_bank},1,200,2,500,USD,0\n"
        + f"2022-01-01 02:00:00,{medium_bank},1,{medium_bank},3,100,USD,0\n",
        encoding="utf-8",
    )
    (raw / "HI-Small_Trans.csv").write_text(
        _ibm_header() + f"2022-02-01 00:00:00,{small_bank},1,200,9,10,USD,0\n",
        encoding="utf-8",
    )


def test_run_feature_build_bank_scoped_directory(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"
    _write_tiny_raw_csvs(raw, medium_bank=100, small_bank=100)
    cfg = load_feature_build_config(default_feature_build_config_path())

    spec = FeatureBuildRunSpec(
        output_subdir=bank_scoped_output_subdir(bank_id=100),
        bank_id=100,
    )
    result = run_feature_build(
        spec,
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        skip_preprocess=False,
        force_rebuild=False,
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )

    proc = feature_output_processed_dir(
        output_subdir="bank_100", processed_root=proc_root
    )
    interim = feature_output_interim_dir(output_subdir="bank_100", interim_root=interim_root)
    paths = feature_parquet_paths(medium_bank_id=100, small_bank_id=100, processed_dir=proc)
    assert paths["experiment_medium"].is_file()
    assert (interim / MANIFEST_JSON).is_file()
    assert result["output_subdir"] == "bank_100"
    assert result["manifest"] == str((interim / MANIFEST_JSON).resolve())


def test_run_feature_build_partial_when_bank_missing_from_one_split(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"
    _write_tiny_raw_csvs(raw, medium_bank=100, small_bank=999)
    cfg = load_feature_build_config(default_feature_build_config_path())

    result = run_feature_build(
        FeatureBuildRunSpec(
            output_subdir=bank_scoped_output_subdir(bank_id=100),
            bank_id=100,
        ),
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        skip_preprocess=False,
        force_rebuild=False,
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )

    proc = feature_output_processed_dir(
        output_subdir="bank_100", processed_root=proc_root
    )
    interim = feature_output_interim_dir(output_subdir="bank_100", interim_root=interim_root)
    paths = feature_parquet_paths(medium_bank_id=100, small_bank_id=100, processed_dir=proc)
    assert result.get("status") != "skipped"
    assert result["available_splits"] == ["medium"]
    assert result["missing_splits"] == ["small"]
    assert paths["experiment_medium"].is_file()
    assert not paths["experiment_small"].is_file()
    assert (interim / MANIFEST_JSON).is_file()


def test_run_feature_build_skips_bank_absent_from_all_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"
    _write_tiny_raw_csvs(raw, medium_bank=999, small_bank=998)
    cfg = load_feature_build_config(default_feature_build_config_path())

    result = run_feature_build(
        FeatureBuildRunSpec(
            output_subdir=bank_scoped_output_subdir(bank_id=100),
            bank_id=100,
        ),
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        skip_preprocess=False,
        force_rebuild=False,
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )

    interim = feature_output_interim_dir(output_subdir="bank_100", interim_root=interim_root)
    assert result["status"] == "skipped"
    assert result["missing_bank_id"] == 100
    assert not (interim / MANIFEST_JSON).exists()


def test_run_feature_build_multiple_banks_isolated(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"
    cfg = load_feature_build_config(default_feature_build_config_path())

    _write_tiny_raw_csvs(raw, medium_bank=100, small_bank=100)
    run_feature_build(
        FeatureBuildRunSpec(
            output_subdir=bank_scoped_output_subdir(bank_id=100), bank_id=100
        ),
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        skip_preprocess=False,
        force_rebuild=False,
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )

    _write_tiny_raw_csvs(raw, medium_bank=70, small_bank=70)
    run_feature_build(
        FeatureBuildRunSpec(
            output_subdir=bank_scoped_output_subdir(bank_id=70), bank_id=70
        ),
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        skip_preprocess=False,
        force_rebuild=False,
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )

    proc_100 = feature_output_processed_dir(
        output_subdir="bank_100", processed_root=proc_root
    )
    proc_70 = feature_output_processed_dir(output_subdir="bank_70", processed_root=proc_root)
    paths_100 = feature_parquet_paths(
        medium_bank_id=100, small_bank_id=100, processed_dir=proc_100
    )
    paths_70 = feature_parquet_paths(
        medium_bank_id=70, small_bank_id=70, processed_dir=proc_70
    )
    exp_100 = pd.read_parquet(paths_100["experiment_medium"])
    exp_70_med = pd.read_parquet(paths_70["experiment_medium"])
    exp_70_sml = pd.read_parquet(paths_70["experiment_small"])
    assert len(exp_100) == 2
    assert len(exp_70_med) == 2
    assert len(exp_70_sml) == 1
    assert not paths_100["experiment_medium"].samefile(paths_70["experiment_medium"])


def test_run_feature_build_split_pair_scoped_directory(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"
    raw.mkdir()
    (raw / "HI-Medium_Trans.csv").write_text(
        _ibm_header() + "2022-01-01 00:00:00,70,1,200,2,500,USD,0\n",
        encoding="utf-8",
    )
    (raw / "HI-Small_Trans.csv").write_text(
        _ibm_header() + "2022-02-01 00:00:00,42,1,200,9,10,USD,0\n",
        encoding="utf-8",
    )
    cfg = load_feature_build_config(default_feature_build_config_path())
    subdir = split_pair_scoped_output_subdir(medium_bank_id=70, small_bank_id=42)

    result = run_feature_build(
        FeatureBuildRunSpec(
            output_subdir=subdir,
            medium_bank_id=70,
            small_bank_id=42,
        ),
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        skip_preprocess=False,
        force_rebuild=False,
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )

    proc = feature_output_processed_dir(output_subdir=subdir, processed_root=proc_root)
    assert result["output_subdir"] == subdir
    assert feature_parquet_paths(
        medium_bank_id=70, small_bank_id=42, processed_dir=proc
    )["txn_level_small"].is_file()


def test_run_feature_build_reuses_existing_artifacts_without_preprocess(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"
    _write_tiny_raw_csvs(raw, medium_bank=100, small_bank=100)
    cfg = load_feature_build_config(default_feature_build_config_path())
    spec = FeatureBuildRunSpec(
        output_subdir=bank_scoped_output_subdir(bank_id=100), bank_id=100
    )
    common = dict(
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        force_rebuild=False,
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )
    first = run_feature_build(spec, skip_preprocess=False, **common)
    assert first.get("reused_existing") is not True

    reused = run_feature_build(spec, skip_preprocess=False, **common)
    assert reused.get("reused_existing") is True
    assert reused["output_subdir"] == "bank_100"
    assert reused["manifest"] == first["manifest"]


def test_run_feature_build_force_rebuild_regenerates(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"
    _write_tiny_raw_csvs(raw, medium_bank=100, small_bank=100)
    cfg = load_feature_build_config(default_feature_build_config_path())
    spec = FeatureBuildRunSpec(
        output_subdir=bank_scoped_output_subdir(bank_id=100), bank_id=100
    )
    common = dict(
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )
    run_feature_build(spec, skip_preprocess=False, force_rebuild=False, **common)
    rebuilt = run_feature_build(spec, skip_preprocess=False, force_rebuild=True, **common)
    assert rebuilt.get("reused_existing") is not True


def test_run_feature_build_skip_preprocess_uses_scoped_cache(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"
    _write_tiny_raw_csvs(raw, medium_bank=100, small_bank=100)
    cfg = load_feature_build_config(default_feature_build_config_path())
    spec = FeatureBuildRunSpec(
        output_subdir=bank_scoped_output_subdir(bank_id=100), bank_id=100
    )
    common = dict(
        raw_dir=raw,
        medium_file="HI-Medium_Trans.csv",
        small_file="HI-Small_Trans.csv",
        force_rebuild=False,
        min_positive=1,
        min_negative=1,
        processed_root=proc_root,
        interim_root=interim_root,
        loaded_feature_cfg=cfg,
    )
    run_feature_build(spec, skip_preprocess=False, **common)
    cached = run_feature_build(spec, skip_preprocess=True, **common)
    proc = feature_output_processed_dir(
        output_subdir="bank_100", processed_root=proc_root
    )
    assert feature_artifacts_present(
        medium_bank_id=100,
        small_bank_id=100,
        processed_dir=proc,
        interim_dir=feature_output_interim_dir(
            output_subdir="bank_100", interim_root=interim_root
        ),
        feature_config_signature=cfg.signature,
    )
    assert cached["output_subdir"] == "bank_100"
