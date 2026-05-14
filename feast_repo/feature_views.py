"""Feature views for account-level daily aggregates (placeholder schema)."""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32, Int64

from data_sources import account_daily_source
from entities import account

account_daily_features = FeatureView(
    name="account_daily_features",
    entities=[account],
    ttl=timedelta(days=365),
    schema=[
        Field(name="daily_tx_count", dtype=Int64),
        Field(name="daily_amount_sum", dtype=Float32),
    ],
    source=account_daily_source,
    online=True,
    offline=True,
)
