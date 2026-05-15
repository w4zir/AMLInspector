"""IBM Kaggle AML CSV column contract and validation.

Dataset layout (typical): duplicate ``Account`` headers become ``Account`` and
``Account.1`` when read with pandas. We normalize to explicit names.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

# Raw CSV column names (Kaggle IBM AML)
COL_TIMESTAMP = "Timestamp"
COL_FROM_BANK = "From Bank"
COL_TO_BANK = "To Bank"
COL_IS_LAUNDERING = "Is Laundering"
# First Account column = sender; second = receiver (often ``Account.1``)
COL_FROM_ACCOUNT_RAW = "Account"
COL_TO_ACCOUNT_SUFFIX = "Account.1"

# Optional (may be absent in subsets)
COL_AMOUNT_RECEIVED = "Amount Received"
COL_RECEIVING_CURRENCY = "Receiving Currency"
COL_AMOUNT_PAID = "Amount Paid"
COL_PAYMENT_CURRENCY = "Payment Currency"
COL_PAYMENT_TYPE = "Payment Type"
COL_PAYMENT_FORMAT = "Payment Format"
COL_SOURCE_COUNTRY = "Source Country"
COL_TARGET_COUNTRY = "Target Country"

REQUIRED_TRANSACTION_COLUMNS: tuple[str, ...] = (
    COL_TIMESTAMP,
    COL_FROM_BANK,
    COL_TO_BANK,
    COL_IS_LAUNDERING,
)

def _read_header_columns(path: Path, *, nrows: int = 0) -> list[str]:
    """Return column names from CSV header."""
    df = pd.read_csv(path, nrows=nrows)
    return list(df.columns)


def validate_raw_csv(path: Path) -> None:
    """Raise ``ValueError`` if required IBM-style columns are missing."""
    cols = set(_read_header_columns(path))
    missing = [c for c in REQUIRED_TRANSACTION_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    has_from = COL_FROM_ACCOUNT_RAW in cols
    # Second account: pandas names duplicate ``Account`` -> ``Account.1``
    if not has_from:
        raise ValueError(f"{path}: missing sender account column {COL_FROM_ACCOUNT_RAW!r}")
    if COL_TO_ACCOUNT_SUFFIX not in cols:
        # Some dumps use unique names; accept any second account column
        account_cols = [c for c in cols if c == COL_FROM_ACCOUNT_RAW or c.startswith("Account")]
        if len(account_cols) < 2:
            raise ValueError(
                f"{path}: expected two account columns (e.g. 'Account' and 'Account.1'); "
                f"found: {account_cols}"
            )


def normalize_account_column_names(columns: Sequence[str]) -> dict[str, str]:
    """Map raw CSV columns to ``from_account_raw`` / ``to_account_raw`` for internal use."""
    cols = list(columns)
    rename: dict[str, str] = {}
    if COL_FROM_ACCOUNT_RAW in cols:
        rename[COL_FROM_ACCOUNT_RAW] = "from_account_raw"
    # Second Account column
    if COL_TO_ACCOUNT_SUFFIX in cols:
        rename[COL_TO_ACCOUNT_SUFFIX] = "to_account_raw"
    else:
        dup = [c for c in cols if c.startswith("Account") and c != COL_FROM_ACCOUNT_RAW]
        if len(dup) >= 1:
            rename[sorted(dup)[0]] = "to_account_raw"
    return rename


def read_csv_chunk(path: Path, *, chunksize: int) -> pd.io.parsers.TextFileReader:
    validate_raw_csv(path)
    return pd.read_csv(path, chunksize=chunksize, low_memory=False)
