"""Load experiment frames from feature-build manifest and Parquet paths."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from aml_inspector.config import DATA_INTERIM, DATA_PROCESSED
from aml_inspector.data.datasets import (
    ENTITY_ID_COL,
    EVENT_TIMESTAMP_COL,
    FEATURE_BASE_ACCOUNT_DAILY,
    FEATURE_BASE_EXPERIMENT,
    LABEL_COL,
    MANIFEST_JSON,
    TRANSACTION_ID_COL,
    bank_scoped_output_subdir,
    dataset_token_for_split,
    feature_output_processed_dir,
    feature_parquet_filename,
    feature_parquet_paths,
)
from aml_inspector.data.preprocess_home_bank import DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL

logger = logging.getLogger(__name__)

DATA_SPLIT_COL = "data_split_source"
SOURCE_BANK_COL = "source_bank_id"

KNOWN_DATASET_SPLITS: tuple[str, ...] = (DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL)


def manifest_path(interim_dir: Path | None = None) -> Path:
    return (interim_dir or DATA_INTERIM) / MANIFEST_JSON


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or manifest_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing feature manifest {p}. "
            "Run: python -m aml_inspector.pipelines.build_feature_data"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_experiment_paths(
    manifest: dict[str, Any],
    *,
    processed_dir: Path | None = None,
) -> tuple[Path, Path, int, int]:
    """Return (medium_experiment_path, small_experiment_path, medium_bank_id, small_bank_id)."""
    medium_id = int(manifest["medium_bank_id"])
    small_id = int(manifest["small_bank_id"])
    outputs = manifest.get("outputs")
    if outputs and "experiment_medium" in outputs and "experiment_small" in outputs:
        return (
            Path(outputs["experiment_medium"]),
            Path(outputs["experiment_small"]),
            medium_id,
            small_id,
        )
    paths = feature_parquet_paths(
        medium_bank_id=medium_id,
        small_bank_id=small_id,
        processed_dir=processed_dir or DATA_PROCESSED,
    )
    return paths["experiment_medium"], paths["experiment_small"], medium_id, small_id


def load_experiment_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment Parquet: {path}")
    return pd.read_parquet(path)


def account_daily_path_from_experiment(experiment_path: Path) -> Path:
    """Derive account_daily Parquet path from an experiment_entity_df path."""
    name = experiment_path.name.replace(
        FEATURE_BASE_EXPERIMENT,
        FEATURE_BASE_ACCOUNT_DAILY,
    )
    return experiment_path.parent / name


def _maybe_join_account_daily(experiment_df: pd.DataFrame, experiment_path: Path) -> pd.DataFrame:
    daily_path = account_daily_path_from_experiment(experiment_path)
    if not daily_path.is_file():
        return experiment_df
    try:
        daily_df = pd.read_parquet(daily_path)
    except OSError:
        return experiment_df
    return join_account_daily_features(experiment_df, daily_df)


def load_medium_small_frames(
    manifest: dict[str, Any] | None = None,
    *,
    manifest_file: Path | None = None,
    processed_dir: Path | None = None,
    join_account_daily: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load HI-Medium and HI-Small experiment entity frames (separate frames)."""
    meta = manifest if manifest is not None else load_manifest(manifest_file)
    med_path, sml_path, _, _ = resolve_experiment_paths(meta, processed_dir=processed_dir)
    medium_df = load_experiment_frame(med_path)
    small_df = load_experiment_frame(sml_path)
    if join_account_daily:
        medium_df = _maybe_join_account_daily(medium_df, med_path)
        small_df = _maybe_join_account_daily(small_df, sml_path)
    _assert_split_tag(medium_df, DATA_SPLIT_MEDIUM)
    _assert_split_tag(small_df, DATA_SPLIT_SMALL)
    return medium_df, small_df, meta


def _bank_split_key(bank_id: int, split: str) -> str:
    return f"{int(bank_id)}:{split}"


def _load_manifest_split_frame(
    experiment_path: Path,
    *,
    split: str,
    join_account_daily: bool,
) -> pd.DataFrame | None:
    if not experiment_path.is_file():
        logger.warning("Skipping manifest %s: missing %s", split, experiment_path)
        return None
    try:
        frame = load_experiment_frame(experiment_path)
    except OSError as e:
        logger.warning("Skipping manifest %s: failed to read %s (%s)", split, experiment_path, e)
        return None
    if join_account_daily:
        frame = _maybe_join_account_daily(frame, experiment_path)
    _assert_split_tag(frame, split)
    return frame


def load_manifest_combined_frame(
    manifest: dict[str, Any] | None = None,
    *,
    manifest_file: Path | None = None,
    processed_dir: Path | None = None,
    join_account_daily: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and concatenate all available HI-Medium and HI-Small frames from a manifest."""
    meta = manifest if manifest is not None else load_manifest(manifest_file)
    med_path, sml_path, _, _ = resolve_experiment_paths(meta, processed_dir=processed_dir)
    frames: list[pd.DataFrame] = []
    for split, path in (
        (DATA_SPLIT_MEDIUM, med_path),
        (DATA_SPLIT_SMALL, sml_path),
    ):
        frame = _load_manifest_split_frame(path, split=split, join_account_daily=join_account_daily)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No HI_MEDIUM or HI_SMALL experiment data found in manifest outputs. "
            f"Checked: {med_path}, {sml_path}"
        )
    return pd.concat(frames, ignore_index=True), meta


def _assert_split_tag(df: pd.DataFrame, expected: str) -> None:
    if DATA_SPLIT_COL not in df.columns:
        return
    actual = df[DATA_SPLIT_COL].astype(str).str.lower().unique().tolist()
    if actual and actual != [expected]:
        raise ValueError(
            f"Expected {DATA_SPLIT_COL}={expected!r}, got {actual!r} in experiment frame"
        )


def manifest_feature_id(manifest: dict[str, Any]) -> str:
    sig = manifest.get("feature_config_signature", "")
    git = manifest.get("git_sha", "unknown")
    return f"{git[:8]}_{sig}" if sig else git[:12]


NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        TRANSACTION_ID_COL,
        EVENT_TIMESTAMP_COL,
        ENTITY_ID_COL,
        DATA_SPLIT_COL,
        LABEL_COL,
        "external_account_hash",
        SOURCE_BANK_COL,
    }
)


@dataclass
class BankFramesLoadResult:
    """Combined experiment frames from one or more bank-scoped processed directories."""

    frame: pd.DataFrame
    requested_bank_ids: list[int] = field(default_factory=list)
    loaded_bank_ids: list[int] = field(default_factory=list)
    skipped_bank_ids: list[int] = field(default_factory=list)
    skip_reasons: dict[int, str] = field(default_factory=dict)
    loaded_bank_splits: list[tuple[int, str]] = field(default_factory=list)
    skipped_bank_splits: list[tuple[int, str]] = field(default_factory=list)
    split_skip_reasons: dict[str, str] = field(default_factory=dict)


def bank_processed_dir(*, bank_id: int, processed_root: Path | None = None) -> Path:
    """``data/processed/bank_<id>/`` (or under ``processed_root``)."""
    return feature_output_processed_dir(
        output_subdir=bank_scoped_output_subdir(bank_id=bank_id),
        processed_root=processed_root or DATA_PROCESSED,
    )


def resolve_bank_split_parquet_paths(
    *,
    bank_id: int,
    split: str,
    processed_root: Path | None = None,
) -> tuple[Path, Path]:
    """Return (experiment_entity_df, account_daily_features) paths for a bank and split."""
    token = dataset_token_for_split(split)
    proc_dir = bank_processed_dir(bank_id=bank_id, processed_root=processed_root)
    experiment = proc_dir / feature_parquet_filename(
        bank_id=bank_id,
        dataset_token=token,
        base_name=FEATURE_BASE_EXPERIMENT,
    )
    account_daily = proc_dir / feature_parquet_filename(
        bank_id=bank_id,
        dataset_token=token,
        base_name=FEATURE_BASE_ACCOUNT_DAILY,
    )
    return experiment, account_daily


def join_account_daily_features(
    experiment_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join daily rollups onto transaction-level experiment rows."""
    out = experiment_df.copy()
    if daily_df.empty or ENTITY_ID_COL not in daily_df.columns:
        out["daily_tx_count"] = pd.Series(0, index=out.index, dtype="int64")
        out["daily_amount_sum"] = pd.Series(0.0, index=out.index, dtype="float64")
        return out

    daily = daily_df.copy()
    daily[EVENT_TIMESTAMP_COL] = pd.to_datetime(daily[EVENT_TIMESTAMP_COL], utc=True)
    out[EVENT_TIMESTAMP_COL] = pd.to_datetime(out[EVENT_TIMESTAMP_COL], utc=True)
    out["_join_day"] = out[EVENT_TIMESTAMP_COL].dt.floor("D")
    daily["_join_day"] = daily[EVENT_TIMESTAMP_COL].dt.floor("D")

    join_keys = [ENTITY_ID_COL, DATA_SPLIT_COL, "_join_day"]
    daily_cols = [ENTITY_ID_COL, DATA_SPLIT_COL, "_join_day", "daily_tx_count", "daily_amount_sum"]
    daily_sub = daily[[c for c in daily_cols if c in daily.columns]].drop_duplicates(
        subset=join_keys
    )

    out = out.merge(
        daily_sub,
        how="left",
        on=join_keys,
        suffixes=("", "_daily_dup"),
    )
    out.drop(columns=["_join_day"], inplace=True, errors="ignore")
    if "daily_tx_count" not in out.columns:
        out["daily_tx_count"] = 0
    if "daily_amount_sum" not in out.columns:
        out["daily_amount_sum"] = 0.0
    out["daily_tx_count"] = pd.to_numeric(out["daily_tx_count"], errors="coerce").fillna(0)
    out["daily_amount_sum"] = pd.to_numeric(out["daily_amount_sum"], errors="coerce").fillna(0.0)
    return out


def _load_single_bank_split_frame(
    *,
    bank_id: int,
    split: str,
    processed_root: Path | None = None,
) -> pd.DataFrame | None:
    """Load one bank's experiment frame with account-daily join; None if files missing."""
    exp_path, daily_path = resolve_bank_split_parquet_paths(
        bank_id=bank_id,
        split=split,
        processed_root=processed_root,
    )
    if not exp_path.is_file():
        logger.warning("Skipping bank %s (%s): missing %s", bank_id, split, exp_path)
        return None
    if not daily_path.is_file():
        logger.warning("Skipping bank %s (%s): missing %s", bank_id, split, daily_path)
        return None
    try:
        experiment_df = pd.read_parquet(exp_path)
        daily_df = pd.read_parquet(daily_path)
    except OSError as e:
        logger.warning("Skipping bank %s (%s): failed to read Parquet (%s)", bank_id, split, e)
        return None

    frame = join_account_daily_features(experiment_df, daily_df)
    frame[SOURCE_BANK_COL] = int(bank_id)
    _assert_split_tag(frame, split)
    return frame


def load_combined_bank_frames(
    bank_ids: list[int],
    *,
    split: str,
    processed_root: Path | None = None,
) -> BankFramesLoadResult:
    """Load and concatenate experiment frames for multiple banks; skip incomplete folders."""
    requested = [int(b) for b in bank_ids]
    loaded: list[int] = []
    skipped: list[int] = []
    reasons: dict[int, str] = {}
    frames: list[pd.DataFrame] = []

    for bank_id in requested:
        frame = _load_single_bank_split_frame(
            bank_id=bank_id,
            split=split,
            processed_root=processed_root,
        )
        if frame is None:
            skipped.append(bank_id)
            exp_path, daily_path = resolve_bank_split_parquet_paths(
                bank_id=bank_id,
                split=split,
                processed_root=processed_root,
            )
            if not exp_path.is_file():
                reasons[bank_id] = f"missing experiment parquet: {exp_path}"
            elif not daily_path.is_file():
                reasons[bank_id] = f"missing account_daily parquet: {daily_path}"
            else:
                reasons[bank_id] = "failed to read parquet"
            continue
        frames.append(frame)
        loaded.append(bank_id)

    if not frames:
        raise FileNotFoundError(
            f"No {split!r} data loaded for bank ids {requested}. "
            f"Skipped: {skipped}. Ensure build_feature_data wrote data/processed/bank_<id>/."
        )

    combined = pd.concat(frames, ignore_index=True)
    return BankFramesLoadResult(
        frame=combined,
        requested_bank_ids=requested,
        loaded_bank_ids=loaded,
        skipped_bank_ids=skipped,
        skip_reasons=reasons,
    )


def load_all_bank_dataset_frames(
    bank_ids: list[int],
    *,
    processed_root: Path | None = None,
) -> BankFramesLoadResult:
    """Load HI-Medium and HI-Small experiment frames for each bank; concatenate all loaded."""
    requested = [int(b) for b in bank_ids]
    frames: list[pd.DataFrame] = []
    loaded_bank_ids: list[int] = []
    skipped_bank_ids: list[int] = []
    bank_reasons: dict[int, str] = {}
    loaded_splits: list[tuple[int, str]] = []
    skipped_splits: list[tuple[int, str]] = []
    split_reasons: dict[str, str] = {}

    for bank_id in requested:
        bank_frames: list[pd.DataFrame] = []
        bank_skip_msgs: list[str] = []
        for split in KNOWN_DATASET_SPLITS:
            frame = _load_single_bank_split_frame(
                bank_id=bank_id,
                split=split,
                processed_root=processed_root,
            )
            if frame is None:
                skipped_splits.append((bank_id, split))
                exp_path, daily_path = resolve_bank_split_parquet_paths(
                    bank_id=bank_id,
                    split=split,
                    processed_root=processed_root,
                )
                if not exp_path.is_file():
                    reason = f"missing experiment parquet: {exp_path}"
                elif not daily_path.is_file():
                    reason = f"missing account_daily parquet: {daily_path}"
                else:
                    reason = "failed to read parquet"
                split_reasons[_bank_split_key(bank_id, split)] = reason
                bank_skip_msgs.append(f"{split}: {reason}")
                continue
            bank_frames.append(frame)
            loaded_splits.append((bank_id, split))

        if bank_frames:
            frames.extend(bank_frames)
            loaded_bank_ids.append(bank_id)
        else:
            skipped_bank_ids.append(bank_id)
            bank_reasons[bank_id] = "; ".join(bank_skip_msgs) if bank_skip_msgs else "no splits loaded"

    if not frames:
        raise FileNotFoundError(
            f"No HI_MEDIUM or HI_SMALL data loaded for bank ids {requested}. "
            f"Banks with no splits: {skipped_bank_ids}. "
            "Ensure build_feature_data wrote data/processed/bank_<id>/."
        )

    combined = pd.concat(frames, ignore_index=True)
    return BankFramesLoadResult(
        frame=combined,
        requested_bank_ids=requested,
        loaded_bank_ids=loaded_bank_ids,
        skipped_bank_ids=skipped_bank_ids,
        skip_reasons=bank_reasons,
        loaded_bank_splits=loaded_splits,
        skipped_bank_splits=skipped_splits,
        split_skip_reasons=split_reasons,
    )


def load_training_bank_frames(
    training_bank_ids: list[int],
    *,
    processed_root: Path | None = None,
) -> BankFramesLoadResult:
    """Combine all available HI-Medium and HI-Small frames for training bank ids."""
    return load_all_bank_dataset_frames(training_bank_ids, processed_root=processed_root)


def load_testing_bank_frames(
    testing_bank_ids: list[int],
    *,
    processed_root: Path | None = None,
) -> BankFramesLoadResult:
    """Combine all available HI-Medium and HI-Small frames for evaluation bank ids."""
    return load_all_bank_dataset_frames(testing_bank_ids, processed_root=processed_root)
