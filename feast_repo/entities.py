"""Feast entities for IBM AML–style modeling (transaction-primary + account aggregates)."""

from feast import Entity

account = Entity(
    name="account",
    join_keys=["account_id"],
    description="Internal account id at Home Bank (after silo masking).",
)

transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    description="Per-transaction modeling row.",
)
