"""Batch sources pointing at project Parquet under data/processed/.

Set ``AML_FEATURE_BANK_ID`` and ``AML_FEATURE_DATASET`` (``HI_MEDIUM`` or ``HI_SMALL``)
to target bank/dataset-specific feature files from ``build_feature_data``.
When unset, falls back to legacy static filenames for local/docker bootstrap stubs.
"""

from __future__ import annotations

import os
from pathlib import Path

from feast import FileSource

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROCESSED = _REPO_ROOT / "data" / "processed"

_LEGACY = {
    "txn_level": "txn_level_features.parquet",
    "account_txn": "account_txn_features.parquet",
    "account_graph": "account_graph_features.parquet",
    "account_daily": "account_daily_features.parquet",
}

_BASE = {
    "txn_level": "txn_level_features",
    "account_txn": "account_txn_features",
    "account_graph": "account_graph_features",
    "account_daily": "account_daily_features",
}


def _feature_parquet_path(table_key: str) -> Path:
    bank_id = os.environ.get("AML_FEATURE_BANK_ID", "").strip()
    dataset = os.environ.get("AML_FEATURE_DATASET", "").strip().upper()
    if bank_id and dataset:
        base = _BASE[table_key]
        name = f"{int(bank_id)}_{dataset}_{base}.parquet"
        return _PROCESSED / name
    return _PROCESSED / _LEGACY[table_key]


txn_level_source = FileSource(
    name="txn_level_source",
    path=str(_feature_parquet_path("txn_level")),
    timestamp_field="event_timestamp",
)

account_txn_source = FileSource(
    name="account_txn_source",
    path=str(_feature_parquet_path("account_txn")),
    timestamp_field="event_timestamp",
)

account_graph_source = FileSource(
    name="account_graph_source",
    path=str(_feature_parquet_path("account_graph")),
    timestamp_field="event_timestamp",
)

account_daily_source = FileSource(
    name="account_daily_source",
    path=str(_feature_parquet_path("account_daily")),
    timestamp_field="event_timestamp",
)
