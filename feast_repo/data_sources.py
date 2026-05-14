"""Batch sources pointing at project Parquet under data/processed/."""

from pathlib import Path

from feast import FileSource

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ACCOUNT_DAILY_PARQUET = _REPO_ROOT / "data" / "processed" / "account_daily_features.parquet"

account_daily_source = FileSource(
    name="account_daily_source",
    path=str(_ACCOUNT_DAILY_PARQUET),
    timestamp_field="event_timestamp",
)
