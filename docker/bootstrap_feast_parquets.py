"""Create minimal Parquet stubs under data/processed when files are missing.

Feast ``apply`` inspects FileSource paths; without files the feature server container exits.
Real data from ``build_feature_data`` replaces these when mounted or copied into the image.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROCESSED = Path("/app/data/processed")

_SPECS: dict[str, dict[str, object]] = {
    "txn_level_features.parquet": {
        "transaction_id": "stub-txn-0",
        "event_timestamp": pd.Timestamp("2020-01-01T00:00:00Z"),
        "data_split_source": "HI-Medium",
        "account_id": "stub-acct-0",
        "from_bank": 1,
        "to_bank": 1,
        "is_outbound": False,
        "counterparty_bank_id": 1,
        "external_account_hash": "stub",
        "amount_usd": 0.0,
        "amount_round_100": False,
        "corridor_risk_score": 0.0,
        "velocity_24h_outbound": 0,
        "dwell_sec_since_last_inbound": 0.0,
        "fanout_unique_internal_7d": 0,
        "fanout_unique_internal_30d": 0,
        "graph_internal_degree": 0,
        "graph_component_size": 0,
        "hrj_country_flag": False,
        "pep_proxy_synthetic": False,
        "company_age_days_proxy": 0.0,
        "label": False,
    },
    "account_txn_features.parquet": {
        "account_id": "stub-acct-0",
        "event_timestamp": pd.Timestamp("2020-01-01T00:00:00Z"),
        "data_split_source": "HI-Medium",
        "velocity_24h_outbound": 0,
        "dwell_sec_since_last_inbound": 0.0,
        "fanout_unique_internal_7d": 0,
        "fanout_unique_internal_30d": 0,
        "company_age_days_proxy": 0.0,
    },
    "account_graph_features.parquet": {
        "account_id": "stub-acct-0",
        "event_timestamp": pd.Timestamp("2020-01-01T00:00:00Z"),
        "graph_internal_degree": 0,
        "graph_component_size": 0,
    },
    "account_daily_features.parquet": {
        "account_id": "stub-acct-0",
        "event_timestamp": pd.Timestamp("2020-01-01T00:00:00Z"),
        "data_split_source": "HI-Medium",
        "daily_tx_count": 0,
        "daily_amount_sum": 0.0,
    },
}


def _needs_write(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return True
    try:
        import pyarrow.parquet as pq

        pq.read_schema(path)
    except Exception:
        return True
    return False


def main() -> None:
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, row in _SPECS.items():
        path = _PROCESSED / name
        if not _needs_write(path):
            continue
        pd.DataFrame([row]).to_parquet(path, index=False)


if __name__ == "__main__":
    main()
