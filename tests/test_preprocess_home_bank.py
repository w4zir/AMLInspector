"""Tests for home-bank preprocessing helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aml_inspector.data.preprocess_home_bank import (
    aggregate_bank_involvement_chunk,
    run_preprocess,
    run_preprocess_medium_small,
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
        "Timestamp,From Bank,Account,To Bank,Account,Is Laundering\n"
        "2022-09-01 00:00:00,100,1,200,2,0\n"
        "2022-09-01 01:00:00,200,2,100,1,1\n"
        "2022-09-01 02:00:00,300,3,400,4,0\n",
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


def test_run_preprocess_medium_small_writes_two_parquets(tmp_path: Path):
    med = tmp_path / "HI-Medium_Trans.csv"
    sml = tmp_path / "HI-Small_Trans.csv"
    body = "Timestamp,From Bank,Account,To Bank,Account,Is Laundering\n2022-01-01,100,1,200,2,0\n"
    med.write_text(body, encoding="utf-8")
    sml.write_text(body, encoding="utf-8")
    out_m = tmp_path / "m.parquet"
    out_s = tmp_path / "s.parquet"
    summary = run_preprocess_medium_small(
        raw_dir=tmp_path,
        bank_id=100,
        medium_file=med.name,
        small_file=sml.name,
        output_medium=out_m,
        output_small=out_s,
        summary_json=tmp_path / "sum.json",
    )
    assert summary["filtered_row_count_medium"] == 1
    assert summary["filtered_row_count_small"] == 1
    mdf = pd.read_parquet(out_m)
    sdf = pd.read_parquet(out_s)
    assert (mdf["data_split_source"] == "medium").all()
    assert (sdf["data_split_source"] == "small").all()
    assert summary["home_bank_id"] == 100
    assert summary["medium_bank_id"] == 100
    assert summary["small_bank_id"] == 100


def test_run_preprocess_medium_small_separate_bank_ids(tmp_path: Path) -> None:
    med = tmp_path / "HI-Medium_Trans.csv"
    sml = tmp_path / "HI-Small_Trans.csv"
    med.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Is Laundering\n"
        "2022-01-01,70,1,200,2,0\n",
        encoding="utf-8",
    )
    sml.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Is Laundering\n"
        "2022-02-01,42,1,200,9,0\n",
        encoding="utf-8",
    )
    summary = run_preprocess_medium_small(
        raw_dir=tmp_path,
        medium_bank_id=70,
        small_bank_id=42,
        medium_file=med.name,
        small_file=sml.name,
        output_medium=tmp_path / "m.parquet",
        output_small=tmp_path / "s.parquet",
        summary_json=tmp_path / "sum.json",
    )
    assert summary["medium_bank_id"] == 70
    assert summary["small_bank_id"] == 42
    assert "home_bank_id" not in summary
    mdf = pd.read_parquet(tmp_path / "m.parquet")
    sdf = pd.read_parquet(tmp_path / "s.parquet")
    assert len(mdf) == 1
    assert len(sdf) == 1


def test_run_preprocess_auto_bank_requires_min_counts(tmp_path: Path):
    csv = tmp_path / "only_neg.csv"
    csv.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Is Laundering\n"
        "2022-09-01 00:00:00,1,10,2,20,0\n",
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
