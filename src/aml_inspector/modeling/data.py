"""Load experiment frames from feature-build manifest and Parquet paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aml_inspector.config import DATA_INTERIM, DATA_PROCESSED
from aml_inspector.data.datasets import (
    ENTITY_ID_COL,
    EVENT_TIMESTAMP_COL,
    LABEL_COL,
    MANIFEST_JSON,
    TRANSACTION_ID_COL,
    feature_parquet_paths,
)
from aml_inspector.data.preprocess_home_bank import DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL

DATA_SPLIT_COL = "data_split_source"


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


def load_medium_small_frames(
    manifest: dict[str, Any] | None = None,
    *,
    manifest_file: Path | None = None,
    processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load HI-Medium and HI-Small experiment entity frames."""
    meta = manifest if manifest is not None else load_manifest(manifest_file)
    med_path, sml_path, _, _ = resolve_experiment_paths(meta, processed_dir=processed_dir)
    medium_df = load_experiment_frame(med_path)
    small_df = load_experiment_frame(sml_path)
    _assert_split_tag(medium_df, DATA_SPLIT_MEDIUM)
    _assert_split_tag(small_df, DATA_SPLIT_SMALL)
    return medium_df, small_df, meta


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
    }
)
