"""Feast entities for IBM AML–style modeling.

Rename `account_id` / join keys after EDA to match your Kaggle CSV columns
(e.g. bank account identifiers). This scaffold keeps one entity for per-account aggregates.
"""

from feast import Entity

account = Entity(
    name="account",
    join_keys=["account_id"],
    description="Placeholder entity for account-level features.",
)
