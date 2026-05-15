"""Feature views aligned to Parquet emitted by aml_inspector.features.build_tables."""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Bool, Float64, Int64, String

from data_sources import (
    account_daily_source,
    account_graph_source,
    account_txn_source,
    txn_level_source,
)
from entities import account, transaction

txn_level_features = FeatureView(
    name="txn_level_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        Field(name="data_split_source", dtype=String),
        Field(name="account_id", dtype=String),
        Field(name="from_bank", dtype=Int64),
        Field(name="to_bank", dtype=Int64),
        Field(name="is_outbound", dtype=Bool),
        Field(name="counterparty_bank_id", dtype=Int64),
        Field(name="external_account_hash", dtype=String),
        Field(name="amount_usd", dtype=Float64),
        Field(name="amount_round_100", dtype=Bool),
        Field(name="corridor_risk_score", dtype=Float64),
        Field(name="velocity_24h_outbound", dtype=Int64),
        Field(name="dwell_sec_since_last_inbound", dtype=Float64),
        Field(name="fanout_unique_internal_7d", dtype=Int64),
        Field(name="fanout_unique_internal_30d", dtype=Int64),
        Field(name="graph_internal_degree", dtype=Int64),
        Field(name="graph_component_size", dtype=Int64),
        Field(name="hrj_country_flag", dtype=Bool),
        Field(name="pep_proxy_synthetic", dtype=Bool),
        Field(name="company_age_days_proxy", dtype=Float64),
        Field(name="label", dtype=Bool),
    ],
    source=txn_level_source,
    online=False,
    offline=True,
)

account_txn_features = FeatureView(
    name="account_txn_features",
    entities=[account],
    ttl=timedelta(days=365),
    schema=[
        Field(name="data_split_source", dtype=String),
        Field(name="velocity_24h_outbound", dtype=Int64),
        Field(name="dwell_sec_since_last_inbound", dtype=Float64),
        Field(name="fanout_unique_internal_7d", dtype=Int64),
        Field(name="fanout_unique_internal_30d", dtype=Int64),
        Field(name="company_age_days_proxy", dtype=Float64),
    ],
    source=account_txn_source,
    online=False,
    offline=True,
)

account_graph_features = FeatureView(
    name="account_graph_features",
    entities=[account],
    ttl=timedelta(days=365),
    schema=[
        Field(name="graph_internal_degree", dtype=Int64),
        Field(name="graph_component_size", dtype=Int64),
    ],
    source=account_graph_source,
    online=False,
    offline=True,
)

account_daily_features = FeatureView(
    name="account_daily_features",
    entities=[account],
    ttl=timedelta(days=365),
    schema=[
        Field(name="data_split_source", dtype=String),
        Field(name="daily_tx_count", dtype=Int64),
        Field(name="daily_amount_sum", dtype=Float64),
    ],
    source=account_daily_source,
    online=False,
    offline=True,
)
