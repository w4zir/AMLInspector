"""Load processed splits and shared column name hints for IBM AML-style tables.

Map your actual Kaggle column names after EDA; placeholders keep imports stable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml_inspector.config import DATA_PROCESSED

# Adjust after inspecting Kaggle CSVs (e.g. account / bank identifiers).
ENTITY_ID_COL = "account_id"
EVENT_TIMESTAMP_COL = "event_timestamp"


def processed_parquet(name: str) -> Path:
    return DATA_PROCESSED / name


def load_parquet(name: str) -> pd.DataFrame:
    path = processed_parquet(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build Parquet from notebooks or ETL, then point Feast at the same path."
        )
    return pd.read_parquet(path)
