"""Load processed splits and shared column names for IBM AML-style tables.

See :mod:`aml_inspector.data.column_contract` for raw CSV column validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml_inspector.config import DATA_PROCESSED

# Canonical modeling columns (after feature pipeline)
ENTITY_ID_COL = "account_id"
TRANSACTION_ID_COL = "transaction_id"
EVENT_TIMESTAMP_COL = "event_timestamp"
LABEL_COL = "label"

# Dataset tokens for bank-prefixed feature Parquet filenames
DATASET_TOKEN_HI_MEDIUM = "HI_MEDIUM"
DATASET_TOKEN_HI_SMALL = "HI_SMALL"

# Processed Parquet filenames (home-bank transaction splits)
PARQUET_HOME_MEDIUM = "home_bank_transactions_medium.parquet"
PARQUET_HOME_SMALL = "home_bank_transactions_small.parquet"

# Legacy static feature filenames (Feast fallback when env vars unset)
PARQUET_TXN_LEVEL = "txn_level_features.parquet"
PARQUET_ACCOUNT_TXN = "account_txn_features.parquet"
PARQUET_ACCOUNT_GRAPH = "account_graph_features.parquet"
PARQUET_ACCOUNT_DAILY = "account_daily_features.parquet"
PARQUET_EXPERIMENT_MEDIUM = "experiment_entity_df_medium.parquet"
PARQUET_EXPERIMENT_SMALL = "experiment_entity_df_small.parquet"

# Feature table base names (suffix after bank/dataset prefix)
FEATURE_BASE_TXN_LEVEL = "txn_level_features"
FEATURE_BASE_ACCOUNT_TXN = "account_txn_features"
FEATURE_BASE_ACCOUNT_GRAPH = "account_graph_features"
FEATURE_BASE_ACCOUNT_DAILY = "account_daily_features"
FEATURE_BASE_EXPERIMENT = "experiment_entity_df"

MANIFEST_JSON = "feature_build_manifest.json"

def dataset_token_for_split(split: str) -> str:
    """Map ``medium`` / ``small`` split labels to filename tokens."""
    s = split.strip().lower()
    if s == "medium":
        return DATASET_TOKEN_HI_MEDIUM
    if s == "small":
        return DATASET_TOKEN_HI_SMALL
    raise ValueError(f"Unknown data split {split!r}; expected 'medium' or 'small'")


def feature_parquet_filename(*, bank_id: int, dataset_token: str, base_name: str) -> str:
    """Build ``{bank_id}_{dataset_token}_{base_name}.parquet``."""
    return f"{int(bank_id)}_{dataset_token}_{base_name}.parquet"


def feature_parquet_paths(
    *,
    medium_bank_id: int,
    small_bank_id: int,
    processed_dir: Path | None = None,
) -> dict[str, Path]:
    """Resolved paths for all bank/dataset-specific feature outputs."""
    root = processed_dir or DATA_PROCESSED
    med_tok = DATASET_TOKEN_HI_MEDIUM
    sml_tok = DATASET_TOKEN_HI_SMALL
    return {
        "txn_level_medium": root
        / feature_parquet_filename(
            bank_id=medium_bank_id,
            dataset_token=med_tok,
            base_name=FEATURE_BASE_TXN_LEVEL,
        ),
        "txn_level_small": root
        / feature_parquet_filename(
            bank_id=small_bank_id, dataset_token=sml_tok, base_name=FEATURE_BASE_TXN_LEVEL
        ),
        "account_txn_medium": root
        / feature_parquet_filename(
            bank_id=medium_bank_id,
            dataset_token=med_tok,
            base_name=FEATURE_BASE_ACCOUNT_TXN,
        ),
        "account_txn_small": root
        / feature_parquet_filename(
            bank_id=small_bank_id,
            dataset_token=sml_tok,
            base_name=FEATURE_BASE_ACCOUNT_TXN,
        ),
        "account_graph_medium": root
        / feature_parquet_filename(
            bank_id=medium_bank_id,
            dataset_token=med_tok,
            base_name=FEATURE_BASE_ACCOUNT_GRAPH,
        ),
        "account_graph_small": root
        / feature_parquet_filename(
            bank_id=small_bank_id,
            dataset_token=sml_tok,
            base_name=FEATURE_BASE_ACCOUNT_GRAPH,
        ),
        "account_daily_medium": root
        / feature_parquet_filename(
            bank_id=medium_bank_id,
            dataset_token=med_tok,
            base_name=FEATURE_BASE_ACCOUNT_DAILY,
        ),
        "account_daily_small": root
        / feature_parquet_filename(
            bank_id=small_bank_id,
            dataset_token=sml_tok,
            base_name=FEATURE_BASE_ACCOUNT_DAILY,
        ),
        "experiment_medium": root
        / feature_parquet_filename(
            bank_id=medium_bank_id,
            dataset_token=med_tok,
            base_name=FEATURE_BASE_EXPERIMENT,
        ),
        "experiment_small": root
        / feature_parquet_filename(
            bank_id=small_bank_id,
            dataset_token=sml_tok,
            base_name=FEATURE_BASE_EXPERIMENT,
        ),
    }


def all_feature_output_filenames(
    *,
    medium_bank_id: int,
    small_bank_id: int,
) -> tuple[str, ...]:
    """Filenames (not paths) required for a complete feature build."""
    return tuple(str(p.name) for p in feature_parquet_paths(
        medium_bank_id=medium_bank_id,
        small_bank_id=small_bank_id,
    ).values())


def processed_parquet(name: str) -> Path:
    return DATA_PROCESSED / name


def load_parquet(name: str) -> pd.DataFrame:
    path = processed_parquet(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m aml_inspector.pipelines.build_feature_data"
        )
    return pd.read_parquet(path)
