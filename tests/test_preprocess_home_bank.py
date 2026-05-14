"""Tests for home-bank preprocessing helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aml_inspector.data.preprocess_home_bank import (
    aggregate_bank_involvement_chunk,
    run_preprocess,
    select_home_bank,
)


def test_aggregate_bank_involvement_dedupes_same_bank_same_row():
    df = pd.DataFrame(
        {
            "From Bank": [10, 20],
            "To Bank": [10, 30],
            "Is Laundering": [1, 0],
        }
    )
    pos, neg = aggregate_bank_involvement_chunk(df)
    # Row0: bank 10 once, laundering -> pos[10]==1
    assert pos[10] == 1
    assert neg[10] == 0
    # Row1: 20 sender neg, 30 receiver neg
    assert neg[20] == 1
    assert neg[30] == 1


def test_select_home_bank_prefers_balance():
    pos = {1: 5, 2: 50}
    neg = {1: 1000, 2: 60}
    assert select_home_bank(pos, neg, min_positive=1, min_negative=1) == 2


def test_run_preprocess_writes_parquet_and_summary(tmp_path: Path):
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "Timestamp,From Bank,To Bank,Is Laundering\n"
        "2022-09-01 00:00:00,100,200,0\n"
        "2022-09-01 01:00:00,200,100,1\n"
        "2022-09-01 02:00:00,300,400,0\n",
        encoding="utf-8",
    )
    out_pq = tmp_path / "out.parquet"
    out_js = tmp_path / "summary.json"
    summary = run_preprocess(
        [str(csv)],
        raw_dir=tmp_path,
        bank_id=100,
        chunksize=10,
        output_parquet=out_pq,
        summary_json=out_js,
        add_event_timestamp=True,
    )
    assert out_pq.is_file()
    assert out_js.is_file()
    assert summary["home_bank_id"] == 100
    assert summary["filtered_row_count"] == 2
    loaded = pd.read_parquet(out_pq)
    assert len(loaded) == 2
    assert "event_timestamp" in loaded.columns
    meta = json.loads(out_js.read_text(encoding="utf-8"))
    assert meta["filtered_row_count"] == 2


def test_run_preprocess_auto_bank_requires_min_counts(tmp_path: Path):
    csv = tmp_path / "only_neg.csv"
    csv.write_text(
        "Timestamp,From Bank,To Bank,Is Laundering\n"
        "2022-09-01 00:00:00,1,2,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No bank satisfies"):
        run_preprocess(
            [str(csv.name)],
            raw_dir=tmp_path,
            bank_id=None,
            min_positive=1,
            min_negative=1,
            output_parquet=tmp_path / "x.parquet",
            summary_json=tmp_path / "y.json",
        )
