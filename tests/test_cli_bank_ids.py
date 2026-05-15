"""Tests for bank id CLI parsing (integers and inclusive ranges)."""

from __future__ import annotations

import argparse
import pytest

from aml_inspector.cli.bank_ids import (
    BankIdListAction,
    parse_bank_id_list,
    parse_bank_id_token,
)


def test_parse_single_bank_id() -> None:
    assert parse_bank_id_token("4") == [4]
    assert parse_bank_id_token(" 70 ") == [70]


def test_parse_inclusive_range() -> None:
    assert parse_bank_id_token("30-33") == [30, 31, 32, 33]
    assert parse_bank_id_token("0-20") == list(range(21))


def test_parse_bank_id_list_mixed() -> None:
    assert parse_bank_id_list(["4", "30-33", "70"]) == [4, 30, 31, 32, 33, 70]
    assert parse_bank_id_list(None) is None


def test_parse_invalid_tokens() -> None:
    with pytest.raises(ValueError, match="invalid bank id token"):
        parse_bank_id_token("abc")
    with pytest.raises(ValueError, match="invalid bank id range"):
        parse_bank_id_token("40-30")
    with pytest.raises(ValueError, match="invalid bank id token"):
        parse_bank_id_token("30--33")


def test_bank_id_list_action() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank-id",
        nargs="+",
        action=BankIdListAction,
        default=None,
    )
    args = parser.parse_args(["--bank-id", "0-2", "5"])
    assert args.bank_id == [0, 1, 2, 5]
