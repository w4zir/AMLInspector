"""Tests for IBM CSV column contract validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aml_inspector.data.column_contract import validate_raw_csv


def test_validate_raw_csv_requires_two_accounts(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "Timestamp,From Bank,To Bank,Is Laundering\n2020-01-01,1,2,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="account"):
        validate_raw_csv(p)


def test_validate_raw_csv_ok_with_duplicate_account_headers(tmp_path: Path):
    p = tmp_path / "ok.csv"
    p.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Is Laundering\n"
        "2020-01-01 00:00:00,1,10,2,20,0\n",
        encoding="utf-8",
    )
    validate_raw_csv(p)
