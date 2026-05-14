"""Filter IBM AML transaction CSVs to a single home bank (sender or receiver).

Reads raw Kaggle-style CSVs in chunks, optionally auto-selects a bank with high
volume of both laundering and non-laundering rows involving that bank, then
writes a Parquet subset and a JSON summary under ``data/processed`` and
``data/interim``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

from aml_inspector.config import DATA_INTERIM, DATA_PROCESSED

COL_FROM_BANK = "From Bank"
COL_TO_BANK = "To Bank"
COL_IS_LAUNDERING = "Is Laundering"
COL_TIMESTAMP = "Timestamp"

DEFAULT_INPUTS = (
    "HI-Small_Trans.csv",
    "HI-Medium_Trans.csv",
)
DEFAULT_PARQUET = "home_bank_transactions.parquet"
DEFAULT_SUMMARY = "home_bank_selection_summary.json"


def _resolve_paths(raw_dir: Path, names: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for n in names:
        p = Path(n)
        if p.is_absolute():
            paths.append(p.resolve())
        elif len(p.parts) == 1:
            paths.append((raw_dir / p).resolve())
        else:
            paths.append(p.resolve())
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(missing))
    return paths


def _coerce_bank_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _normalize_label(series: pd.Series) -> pd.Series:
    """Coerce laundering column to boolean (handles 0/1, true/false strings)."""
    if series.dtype == object:
        lower = series.astype(str).str.strip().str.lower()
        return lower.isin(("1", "true", "yes", "y"))
    return series.astype("Int64").fillna(0).astype(int).astype(bool)


def aggregate_bank_involvement_chunk(
    df: pd.DataFrame,
    *,
    col_from: str = COL_FROM_BANK,
    col_to: str = COL_TO_BANK,
    col_label: str = COL_IS_LAUNDERING,
) -> tuple[Counter[int], Counter[int]]:
    """Return positive and negative per-bank involvement counts for one chunk.

    Each row contributes at most once per bank per side; if From and To are
    the same bank, that bank gets a single increment for that row.
    """
    y = _normalize_label(df[col_label])
    idx = df.index
    p1 = pd.DataFrame({"idx": idx, "bank": _coerce_bank_series(df[col_from]), "_pos": y})
    p2 = pd.DataFrame({"idx": idx, "bank": _coerce_bank_series(df[col_to]), "_pos": y})
    long = pd.concat([p1, p2], ignore_index=True)
    long = long.dropna(subset=["bank"])
    long = long.drop_duplicates(subset=["idx", "bank"])

    pos_c: Counter[int] = Counter()
    neg_c: Counter[int] = Counter()
    for bank, g in long.groupby("bank", sort=False):
        b_int = int(bank)
        pos = int(g["_pos"].sum())
        neg = int((~g["_pos"]).sum())
        if pos:
            pos_c[b_int] += pos
        if neg:
            neg_c[b_int] += neg
    return pos_c, neg_c


def merge_counters(a: Counter[int], b: Counter[int]) -> Counter[int]:
    out = Counter(a)
    out.update(b)
    return out


def select_home_bank(
    pos: Counter[int],
    neg: Counter[int],
    *,
    min_positive: int = 1,
    min_negative: int = 1,
) -> int:
    """Pick bank maximizing balanced volume: min(pos, neg), then total."""
    candidates: list[tuple[int, int, int, int]] = []
    banks = set(pos.keys()) | set(neg.keys())
    for b in banks:
        p, n = pos.get(b, 0), neg.get(b, 0)
        if p < min_positive or n < min_negative:
            continue
        balance = min(p, n)
        total = p + n
        candidates.append((balance, total, p, b))
    if not candidates:
        raise ValueError(
            "No bank satisfies min_positive/min_negative thresholds. "
            f"Try lowering thresholds or using a larger input (pos banks: {len(pos)}, neg banks: {len(neg)})."
        )
    candidates.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))
    return candidates[0][3]


def scan_banks_from_csvs(
    paths: list[Path],
    *,
    chunksize: int = 200_000,
    min_positive: int = 1,
    min_negative: int = 1,
) -> tuple[int, Counter[int], Counter[int]]:
    pos_total: Counter[int] = Counter()
    neg_total: Counter[int] = Counter()
    for path in paths:
        reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            p, n = aggregate_bank_involvement_chunk(chunk)
            pos_total = merge_counters(pos_total, p)
            neg_total = merge_counters(neg_total, n)
    bank_id = select_home_bank(
        pos_total,
        neg_total,
        min_positive=min_positive,
        min_negative=min_negative,
    )
    return bank_id, pos_total, neg_total


def filter_csvs_to_parquet(
    paths: list[Path],
    bank_id: int,
    output_parquet: Path,
    *,
    chunksize: int = 200_000,
    add_event_timestamp: bool = True,
) -> int:
    """Stream-filter rows to Parquet; returns total row count written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    try:
        for path in paths:
            reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                f_b = _coerce_bank_series(chunk[COL_FROM_BANK])
                t_b = _coerce_bank_series(chunk[COL_TO_BANK])
                mask = ((f_b == bank_id) | (t_b == bank_id)).fillna(False)
                sub = chunk.loc[mask]
                if sub.empty:
                    continue
                if add_event_timestamp and COL_TIMESTAMP in sub.columns:
                    sub = sub.copy()
                    sub["event_timestamp"] = pd.to_datetime(
                        sub[COL_TIMESTAMP], errors="coerce", utc=True
                    )
                table = pa.Table.from_pandas(sub, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_parquet), table.schema)
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                total_rows += len(sub)
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        output_parquet.unlink(missing_ok=True)
        raise ValueError(
            f"No rows matched bank_id={bank_id} after filtering. Check column types (integer bank ids)."
        )

    return total_rows


def run_preprocess(
    input_files: list[str | Path],
    *,
    raw_dir: Path | None = None,
    bank_id: int | None = None,
    chunksize: int = 200_000,
    min_positive: int = 1,
    min_negative: int = 1,
    output_parquet: Path | None = None,
    summary_json: Path | None = None,
    add_event_timestamp: bool = True,
) -> dict:
    from aml_inspector.config import DATA_RAW

    raw = raw_dir if raw_dir is not None else DATA_RAW
    paths = _resolve_paths(raw, (str(p) for p in input_files))

    if bank_id is None:
        selected, pos_c, neg_c = scan_banks_from_csvs(
            paths,
            chunksize=chunksize,
            min_positive=min_positive,
            min_negative=min_negative,
        )
    else:
        selected = int(bank_id)
        pos_c, neg_c = Counter(), Counter()
        for path in paths:
            reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                p, n = aggregate_bank_involvement_chunk(chunk)
                pos_c = merge_counters(pos_c, p)
                neg_c = merge_counters(neg_c, n)

    out_pq = output_parquet or (DATA_PROCESSED / DEFAULT_PARQUET)
    out_json = summary_json or (DATA_INTERIM / DEFAULT_SUMMARY)

    n_written = filter_csvs_to_parquet(
        paths,
        selected,
        out_pq,
        chunksize=chunksize,
        add_event_timestamp=add_event_timestamp,
    )

    pos_n = int(pos_c.get(selected, 0))
    neg_n = int(neg_c.get(selected, 0))

    summary = {
        "home_bank_id": selected,
        "positive_involvement_count": pos_n,
        "negative_involvement_count": neg_n,
        "filtered_row_count": n_written,
        "input_files": [str(p) for p in paths],
        "output_parquet": str(out_pq.resolve()),
        "chunksize": chunksize,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Select a home bank and filter IBM AML CSVs to sender/receiver rows.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="CSV paths or basenames under data/raw (default: HI-Small_Trans.csv HI-Medium_Trans.csv)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory to resolve bare filenames (default: project data/raw)",
    )
    parser.add_argument(
        "--bank-id",
        type=int,
        default=None,
        help="Force home bank id (skip auto selection)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Pandas read_csv chunksize",
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=1,
        help="Minimum laundering involvement count for auto bank pick",
    )
    parser.add_argument(
        "--min-negative",
        type=int,
        default=1,
        help="Minimum non-laundering involvement count for auto bank pick",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=None,
        help=f"Output Parquet path (default: data/processed/{DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=f"Summary JSON path (default: data/interim/{DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--no-event-timestamp",
        action="store_true",
        help="Do not add event_timestamp from Timestamp column",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_preprocess(
            list(args.input_files),
            raw_dir=args.raw_dir,
            bank_id=args.bank_id,
            chunksize=args.chunksize,
            min_positive=args.min_positive,
            min_negative=args.min_negative,
            output_parquet=args.output_parquet,
            summary_json=args.summary_json,
            add_event_timestamp=not args.no_event_timestamp,
        )
    except (FileNotFoundError, ValueError) as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
