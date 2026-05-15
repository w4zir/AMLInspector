"""Build point-in-time-safe Parquet feature tables from home-bank Medium/Small subsets.

Medium is used to fit frozen artifacts (corridor normalization, weekly internal graph
snapshots). Small rows only consume those artifacts — no Small statistics are used
for fitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from aml_inspector.config import DATA_INTERIM, DATA_PROCESSED, PROJECT_ROOT
from aml_inspector.data.column_contract import (
    COL_AMOUNT_PAID,
    COL_AMOUNT_RECEIVED,
    COL_FROM_BANK,
    COL_IS_LAUNDERING,
    COL_PAYMENT_CURRENCY,
    COL_RECEIVING_CURRENCY,
    COL_SOURCE_COUNTRY,
    COL_TARGET_COUNTRY,
    COL_TIMESTAMP,
    COL_TO_BANK,
    normalize_account_column_names,
)
from aml_inspector.data.datasets import (
    ENTITY_ID_COL,
    EVENT_TIMESTAMP_COL,
    LABEL_COL,
    MANIFEST_JSON,
    TRANSACTION_ID_COL,
    feature_parquet_paths,
)
from aml_inspector.data.preprocess_home_bank import DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL
from aml_inspector.features.feature_build_config import (
    FeatureBuildFlags,
    LoadedFeatureBuildConfig,
    default_feature_build_config_path,
    load_feature_build_config,
)

logger = logging.getLogger(__name__)

SALT_FILENAME = "project_hash_salt.txt"
FX_FILENAME = "fx_rates_mvp.json"
CORRIDOR_FIT_FILENAME = "corridor_risk_fit_medium.json"

_MEDIUM_OUTPUT_KEYS = (
    "txn_level_medium",
    "account_txn_medium",
    "account_graph_medium",
    "account_daily_medium",
    "experiment_medium",
)
_SMALL_OUTPUT_KEYS = (
    "txn_level_small",
    "account_txn_small",
    "account_graph_small",
    "account_daily_small",
    "experiment_small",
)


def _resolve_available_splits(
    path_medium: Path,
    path_small: Path,
    preprocess_summary: dict[str, Any] | None,
) -> set[str]:
    if preprocess_summary is not None and "available_splits" in preprocess_summary:
        return {str(s) for s in preprocess_summary["available_splits"]}
    available: set[str] = set()
    if path_medium.is_file():
        available.add(DATA_SPLIT_MEDIUM)
    if path_small.is_file():
        available.add(DATA_SPLIT_SMALL)
    return available


def feature_artifacts_present(
    *,
    medium_bank_id: int,
    small_bank_id: int,
    processed_dir: Path,
    interim_dir: Path,
    feature_config_signature: str | None = None,
) -> dict[str, Any] | None:
    """If manifest matches bank ids and all bank/dataset feature Parquets exist, return manifest."""
    manifest_path = interim_dir / MANIFEST_JSON
    if not manifest_path.is_file():
        return None
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if int(manifest.get("medium_bank_id", -1)) != int(medium_bank_id):
        return None
    if int(manifest.get("small_bank_id", -1)) != int(small_bank_id):
        return None
    if feature_config_signature is not None:
        if manifest.get("feature_config_signature") != feature_config_signature:
            return None
    paths = feature_parquet_paths(
        medium_bank_id=medium_bank_id,
        small_bank_id=small_bank_id,
        processed_dir=processed_dir,
    )
    available = manifest.get("available_splits", [DATA_SPLIT_MEDIUM, DATA_SPLIT_SMALL])
    keys_to_check: list[str] = []
    if DATA_SPLIT_MEDIUM in available:
        keys_to_check.extend(_MEDIUM_OUTPUT_KEYS)
    if DATA_SPLIT_SMALL in available:
        keys_to_check.extend(_SMALL_OUTPUT_KEYS)
    for key in keys_to_check:
        if not paths[key].is_file():
            return None
    return manifest


def _filter_home_bank_rows(
    df: pd.DataFrame,
    home_bank_id: int,
    *,
    split_label: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep rows where the home bank is sender or receiver (defensive after load)."""
    fb = pd.to_numeric(df[COL_FROM_BANK], errors="coerce")
    tb = pd.to_numeric(df[COL_TO_BANK], errors="coerce")
    home = int(home_bank_id)
    n_before = int(len(df))
    mask = ((fb == home) | (tb == home)).fillna(False)
    out = df.loc[mask].copy()
    n_after = int(len(out))
    logger.info(
        "home_bank filter (%s): rows %s -> %s for home_bank_id=%s",
        split_label,
        n_before,
        n_after,
        home,
    )
    if n_after == 0:
        raise ValueError(
            f"No rows left after home-bank filter for {split_label} (home_bank_id={home}). "
            "Re-run preprocess or check Parquet contents."
        )
    stats = {"split": split_label, "rows_before": n_before, "rows_after": n_after}
    return out, stats


def _event_timestamp_ns_numpy(series: pd.Series) -> np.ndarray:
    """Int64 UTC nanoseconds for vectorized time-window logic (replaces deprecated ``.view('int64')``)."""
    return np.asarray(pd.DatetimeIndex(series).asi8, dtype=np.int64)


def _log_group_loop_progress(phase: str, index: int, total: int) -> None:
    if total <= 0:
        return
    step = max(1, total // 20)
    if index == 1 or index == total or index % step == 0:
        logger.info("%s: processed %s / %s account-split groups (~%.0f%%)", phase, index, total, 100.0 * index / total)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def ensure_hash_salt(interim_dir: Path) -> str:
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / SALT_FILENAME
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    import secrets

    salt = secrets.token_hex(32)
    path.write_text(salt, encoding="utf-8")
    return salt


def default_fx_rates() -> dict[str, float]:
    """Fixed MVP rates vs USD (documented; not historical FX)."""
    return {
        "USD": 1.0,
        "US dollar": 1.0,
        "EUR": 1.08,
        "GBP": 1.27,
        "JPY": 0.0067,
        "CNY": 0.14,
        "INR": 0.012,
        "CAD": 0.74,
        "AUD": 0.66,
        "CHF": 1.12,
        "MXN": 0.059,
        "UK pounds": 1.27,
        "EURO": 1.08,
        "YEN": 0.0067,
        "US Dollar": 1.0,
        "Rupee": 0.012,
        "Yuan": 0.14,
        "Canadian Dollar": 0.74,
        "Australian Dollar": 0.66,
        "Swiss Franc": 1.12,
        "Mexican Peso": 0.059,
    }


def load_or_create_fx(interim_dir: Path) -> dict[str, float]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / FX_FILENAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    rates = default_fx_rates()
    path.write_text(json.dumps(rates, indent=2), encoding="utf-8")
    return rates


def _stable_series_hash(values: pd.Series, salt: str) -> pd.Series:
    out: list[str] = []
    for v in values.astype(str):
        h = hashlib.sha256(f"{salt}:{v}".encode()).hexdigest()[:16]
        out.append(h)
    return pd.Series(out, index=values.index, dtype="string")


def _rename_country_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if COL_SOURCE_COUNTRY in df.columns:
        df["_src_country"] = df[COL_SOURCE_COUNTRY].astype(str).str.strip()
    else:
        df["_src_country"] = ""
    if COL_TARGET_COUNTRY in df.columns:
        df["_tgt_country"] = df[COL_TARGET_COUNTRY].astype(str).str.strip()
    else:
        df["_tgt_country"] = ""
    return df


def fit_corridor_scores(medium_df: pd.DataFrame, interim_dir: Path) -> dict[str, Any]:
    """Log-count corridor weights from Medium only (frozen)."""
    m = _rename_country_columns(medium_df)
    mask = (m["_src_country"] != "") & (m["_tgt_country"] != "") & (m["_src_country"] != "nan")
    sub = m.loc[mask, ["_src_country", "_tgt_country"]]
    counts = sub.groupby(["_src_country", "_tgt_country"], observed=False).size()
    if len(counts) == 0:
        weights: dict[str, float] = {}
        max_c = 1.0
    else:
        max_c = float(counts.max())
        weights = {f"{a}||{b}": float(np.log1p(c) / np.log1p(max_c)) for (a, b), c in counts.items()}
    payload = {"weights": weights, "max_count": max_c}
    interim_dir.mkdir(parents=True, exist_ok=True)
    (interim_dir / CORRIDOR_FIT_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def apply_corridor_scores(df: pd.DataFrame, fit: dict[str, Any]) -> pd.Series:
    m = _rename_country_columns(df)
    w = fit.get("weights", {})
    keys = m["_src_country"] + "||" + m["_tgt_country"]
    return keys.map(lambda k: float(w.get(k, 0.0))).astype("float64")


def _hrj_flag_for_counterparty_leg(df: pd.DataFrame, home_bank_id: int) -> pd.Series:
    """MVP: high-risk if counterparty *country* string matches grey-list tokens."""
    grey = ("iran", "north korea", "myanmar", "syria", "afghanistan", "yemen", "cuba")
    fb = pd.to_numeric(df[COL_FROM_BANK], errors="coerce")
    # country on counterparty leg
    cc = np.where(fb == home_bank_id, df["_tgt_country"], df["_src_country"])
    s = pd.Series(cc, index=df.index, dtype="string").str.lower()
    return s.str.contains("|".join(grey), regex=True).fillna(False)


def apply_column_normalization(df: pd.DataFrame) -> pd.DataFrame:
    ren = normalize_account_column_names(df.columns)
    return df.rename(columns=ren)


def enrich_transaction_frame(
    df: pd.DataFrame,
    home_bank_id: int,
    salt: str,
    fx: dict[str, float],
) -> pd.DataFrame:
    df = apply_column_normalization(df)
    if EVENT_TIMESTAMP_COL in df.columns and df[EVENT_TIMESTAMP_COL].notna().any():
        df[EVENT_TIMESTAMP_COL] = pd.to_datetime(
            df[EVENT_TIMESTAMP_COL], utc=True, errors="coerce"
        )
    else:
        df[EVENT_TIMESTAMP_COL] = pd.to_datetime(df[COL_TIMESTAMP], utc=True, errors="coerce")
    df = _rename_country_columns(df)
    fb = pd.to_numeric(df[COL_FROM_BANK], errors="coerce")
    tb = pd.to_numeric(df[COL_TO_BANK], errors="coerce")
    home = int(home_bank_id)

    df["internal_account_id"] = np.where(fb == home, df["from_account_raw"], df["to_account_raw"]).astype(str)
    df["is_outbound"] = (fb == home).fillna(False)
    df["counterparty_bank_id"] = pd.Series(np.where(fb == home, tb, fb), dtype="Int64")

    both_home = (fb == home) & (tb == home)
    ext_raw = np.where(fb == home, df["to_account_raw"], df["from_account_raw"])
    df["external_account_hash"] = ""
    mask_ext = ~both_home.fillna(False)
    if mask_ext.any():
        df.loc[mask_ext, "external_account_hash"] = _stable_series_hash(
            pd.Series(ext_raw, index=df.index).loc[mask_ext],
            salt,
        )

    # Amount USD (MVP)
    def _to_usd(row: pd.Series) -> float:
        if COL_AMOUNT_PAID in row.index and pd.notna(row.get(COL_AMOUNT_PAID)):
            amt = float(pd.to_numeric(row[COL_AMOUNT_PAID], errors="coerce") or 0.0)
            cur = str(row.get(COL_PAYMENT_CURRENCY, "USD") or "USD").strip()
        elif COL_AMOUNT_RECEIVED in row.index and pd.notna(row.get(COL_AMOUNT_RECEIVED)):
            amt = float(pd.to_numeric(row[COL_AMOUNT_RECEIVED], errors="coerce") or 0.0)
            cur = str(row.get(COL_RECEIVING_CURRENCY, "USD") or "USD").strip()
        else:
            return 0.0
        rate = float(fx.get(cur, fx.get(cur.upper(), 1.0)))
        return amt * rate

    logger.info("Computing amount_usd for %s rows …", len(df))
    df["amount_usd"] = df.apply(_to_usd, axis=1)
    df["amount_round_100"] = (df["amount_usd"] % 100.0 < 1e-6) & (df["amount_usd"] > 0)

    # Transaction id (stable given row fields)
    hsrc = (
        df[COL_FROM_BANK].astype(str)
        + "|"
        + df["from_account_raw"].astype(str)
        + "|"
        + df[COL_TO_BANK].astype(str)
        + "|"
        + df["to_account_raw"].astype(str)
        + "|"
        + df[EVENT_TIMESTAMP_COL].astype(str)
    )
    df[TRANSACTION_ID_COL] = [
        hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()[:24] for s in hsrc
    ]

    # Label
    if COL_IS_LAUNDERING in df.columns:
        y = df[COL_IS_LAUNDERING]
        if y.dtype == object:
            low = y.astype(str).str.strip().str.lower()
            df[LABEL_COL] = low.isin(("1", "true", "yes", "y"))
        else:
            df[LABEL_COL] = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).astype(bool)
    else:
        df[LABEL_COL] = False

    df[ENTITY_ID_COL] = df["internal_account_id"]
    df["from_bank"] = fb.astype("Int64")
    df["to_bank"] = tb.astype("Int64")
    return df


def _velocity_24h_outbound(df: pd.DataFrame) -> pd.Series:
    """Strictly prior 24h outbound count per (account, split)."""
    out = pd.Series(0, index=df.index, dtype="int64")
    g_sorted = df.sort_values([ENTITY_ID_COL, "data_split_source", EVENT_TIMESTAMP_COL])
    window_ns = int(pd.Timedelta(hours=24).value)
    gb = g_sorted.groupby([ENTITY_ID_COL, "data_split_source"], sort=False)
    total_groups = gb.ngroups
    logger.info("velocity_24h_outbound: %s rows, %s account-split groups", len(g_sorted), total_groups)
    for gi, ((_, _split), g) in enumerate(gb, start=1):
        _log_group_loop_progress("velocity_24h_outbound", gi, total_groups)
        ts = _event_timestamp_ns_numpy(g[EVENT_TIMESTAMP_COL])
        ob = g["is_outbound"].to_numpy(dtype=bool)
        idxs = g.index.to_numpy()
        n = len(g)
        j = 0
        for i in range(n):
            t_i = ts[i]
            lo = t_i - window_ns
            while j < i and ts[j] < lo:
                j += 1
            acc = 0
            for k in range(j, i):
                if ts[k] < t_i and ob[k]:
                    acc += 1
            out.loc[idxs[i]] = acc
    return out


def _dwell_since_last_inbound(df: pd.DataFrame) -> pd.Series:
    """Seconds since last inbound before this row (inbound rows update state first)."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    g_sorted = df.sort_values(
        [ENTITY_ID_COL, "data_split_source", EVENT_TIMESTAMP_COL, "is_outbound"],
    )
    gb = g_sorted.groupby([ENTITY_ID_COL, "data_split_source"], sort=False)
    total_groups = gb.ngroups
    logger.info("dwell_sec_since_last_inbound: %s rows, %s account-split groups", len(g_sorted), total_groups)
    for gi, ((_, _split), g) in enumerate(gb, start=1):
        _log_group_loop_progress("dwell_sec_since_last_inbound", gi, total_groups)
        last_in = pd.NaT
        for idx, row in g.iterrows():
            ts = row[EVENT_TIMESTAMP_COL]
            is_out = bool(row["is_outbound"])
            if not is_out and pd.notna(ts):
                last_in = ts
            if is_out and pd.notna(ts) and pd.notna(last_in):
                out.loc[idx] = (ts - last_in) / pd.Timedelta(seconds=1)
    return out


def _rolling_unique_internal_recipients(
    df: pd.DataFrame, window_days: int, home_bank_id: int
) -> pd.Series:
    """Count distinct internal counterparties on strictly prior rows in window (same split)."""
    fb = pd.to_numeric(df[COL_FROM_BANK], errors="coerce")
    tb = pd.to_numeric(df[COL_TO_BANK], errors="coerce")
    internal_edge = (fb == home_bank_id) & (tb == home_bank_id)
    cp = np.where(fb == home_bank_id, df["to_account_raw"], df["from_account_raw"])
    cp = pd.Series(cp, index=df.index).where(internal_edge.to_numpy())

    out = pd.Series(0, index=df.index, dtype="int64")
    g_sorted = df.sort_values([ENTITY_ID_COL, "data_split_source", EVENT_TIMESTAMP_COL])
    window_ns = int(pd.Timedelta(days=window_days).value)

    gb = g_sorted.groupby([ENTITY_ID_COL, "data_split_source"], sort=False)
    total_groups = gb.ngroups
    logger.info(
        "fanout_unique_internal_%sd: %s rows, %s account-split groups",
        window_days,
        len(g_sorted),
        total_groups,
    )
    for gi, ((_, _split), g) in enumerate(gb, start=1):
        _log_group_loop_progress(f"fanout_unique_internal_{window_days}d", gi, total_groups)
        t_int = _event_timestamp_ns_numpy(g[EVENT_TIMESTAMP_COL])
        partners = cp.reindex(g.index).to_numpy()
        idxs = g.index.to_numpy()
        n = len(g)
        for i in range(n):
            t_i = t_int[i]
            lo = t_i - window_ns
            seen: set[str] = set()
            for k in range(i):
                if t_int[k] < t_i and lo <= t_int[k]:
                    pk = partners[k]
                    if pk is not None and not (isinstance(pk, float) and np.isnan(pk)):
                        s = str(pk)
                        if s and s != "<NA>":
                            seen.add(s)
            out.loc[idxs[i]] = len(seen)
    return out


def _company_age_days(df: pd.DataFrame) -> pd.Series:
    """Days since first strictly prior event for same (account, split)."""
    df = df.sort_values([ENTITY_ID_COL, "data_split_source", EVENT_TIMESTAMP_COL])
    first = df.groupby([ENTITY_ID_COL, "data_split_source"], sort=False)[EVENT_TIMESTAMP_COL].cummin()
    age = (df[EVENT_TIMESTAMP_COL] - first) / pd.Timedelta(days=1)
    return age.astype("float64")


def _pep_proxy_synthetic(account: pd.Series, salt: str) -> pd.Series:
    out = []
    for a in account.astype(str):
        h = int(hashlib.sha256(f"{salt}:pep:{a}".encode()).hexdigest(), 16)
        out.append(bool(h % 503 == 0))
    return pd.Series(out, index=account.index, dtype=bool)


def build_weekly_internal_graph_features(
    medium_df: pd.DataFrame,
    home_bank_id: int,
) -> pd.DataFrame:
    """One row per (iso_week period start, account_id) from **previous** week's internal graph."""
    fb = pd.to_numeric(medium_df[COL_FROM_BANK], errors="coerce")
    tb = pd.to_numeric(medium_df[COL_TO_BANK], errors="coerce")
    home = int(home_bank_id)
    sub = medium_df.loc[(fb == home) & (tb == home)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["period", ENTITY_ID_COL, "graph_internal_degree", "graph_component_size"]
        )
    sub[EVENT_TIMESTAMP_COL] = pd.to_datetime(sub[EVENT_TIMESTAMP_COL], utc=True)
    ts = sub[EVENT_TIMESTAMP_COL]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    sub["_period"] = ts.dt.to_period("W-MON")

    rows: list[dict[str, Any]] = []
    periods = sorted(sub["_period"].unique())
    n_periods = len(periods)
    work_weeks = max(0, n_periods - 1)
    logger.info(
        "weekly_internal_graph: %s internal-edge rows, %s iso-week buckets, %s prior-week graph builds",
        len(sub),
        n_periods,
        work_weeks,
    )
    for i, p in enumerate(periods):
        if i == 0:
            continue
        prev_p = periods[i - 1]
        edges_df = sub.loc[sub["_period"] == prev_p, ["from_account_raw", "to_account_raw"]]
        if work_weeks > 0:
            slot = i
            step = max(1, work_weeks // 20)
            if slot == 1 or slot == n_periods - 1 or slot % step == 0:
                logger.info(
                    "weekly_internal_graph: week slot %s / %s (%s prior-week edges)",
                    slot,
                    work_weeks,
                    len(edges_df),
                )
        g = nx.Graph()
        for a, b in edges_df.itertuples(index=False, name=None):
            aa, bb = str(a), str(b)
            if aa != bb:
                g.add_edge(aa, bb)
        comps = list(nx.connected_components(g)) if len(g) else []
        comp_size = {n: len(c) for c in comps for n in c}
        deg = dict(g.degree())
        for node in g.nodes():
            rows.append(
                {
                    "period": str(p),
                    ENTITY_ID_COL: node,
                    "graph_internal_degree": int(deg.get(node, 0)),
                    "graph_component_size": int(comp_size.get(node, 1)),
                }
            )
    cols = ["period", ENTITY_ID_COL, "graph_internal_degree", "graph_component_size"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def build_account_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_d"] = df[EVENT_TIMESTAMP_COL].dt.floor("D")
    g = df.groupby([ENTITY_ID_COL, "data_split_source", "_d"], as_index=False).agg(
        daily_tx_count=(TRANSACTION_ID_COL, "count"),
        daily_amount_sum=("amount_usd", "sum"),
    )
    g = g.rename(columns={"_d": EVENT_TIMESTAMP_COL})
    return g


def _empty_account_daily_frame() -> pd.DataFrame:
    """Schema-compatible empty frame when account_daily is disabled."""
    return pd.DataFrame(
        {
            ENTITY_ID_COL: pd.Series(dtype="string"),
            EVENT_TIMESTAMP_COL: pd.Series(dtype="datetime64[ns, UTC]"),
            "data_split_source": pd.Series(dtype="string"),
            "daily_tx_count": pd.Series(dtype="int64"),
            "daily_amount_sum": pd.Series(dtype="float64"),
        }
    )


def _apply_rolling_and_proxy_features(
    df: pd.DataFrame,
    *,
    bank_id: int,
    salt: str,
    flags: FeatureBuildFlags,
) -> pd.DataFrame:
    """Point-in-time rolling and proxy columns for one split's frame."""
    out = df.copy()
    if flags.rolling_account_activity:
        out["velocity_24h_outbound"] = _velocity_24h_outbound(out).astype("Int64")
        out["dwell_sec_since_last_inbound"] = _dwell_since_last_inbound(out).astype("float64")
        out["fanout_unique_internal_7d"] = _rolling_unique_internal_recipients(
            out, 7, bank_id
        ).astype("Int64")
        out["fanout_unique_internal_30d"] = _rolling_unique_internal_recipients(
            out, 30, bank_id
        ).astype("Int64")
    else:
        out["velocity_24h_outbound"] = pd.Series(0, index=out.index, dtype="Int64")
        out["dwell_sec_since_last_inbound"] = pd.Series(np.nan, index=out.index, dtype="float64")
        out["fanout_unique_internal_7d"] = pd.Series(0, index=out.index, dtype="Int64")
        out["fanout_unique_internal_30d"] = pd.Series(0, index=out.index, dtype="Int64")

    if flags.company_age_days_proxy:
        out["company_age_days_proxy"] = _company_age_days(out).astype("float64")
    else:
        out["company_age_days_proxy"] = pd.Series(0.0, index=out.index, dtype="float64")

    if flags.pep_proxy_synthetic:
        out["pep_proxy_synthetic"] = _pep_proxy_synthetic(out[ENTITY_ID_COL], salt)
    else:
        out["pep_proxy_synthetic"] = pd.Series(False, index=out.index, dtype=bool)

    if flags.hrj_country_flag:
        out["hrj_country_flag"] = _hrj_flag_for_counterparty_leg(out, bank_id)
    else:
        out["hrj_country_flag"] = pd.Series(False, index=out.index, dtype=bool)
    return out


def build_all_feature_tables(
    *,
    medium_bank_id: int,
    small_bank_id: int,
    path_medium: Path,
    path_small: Path,
    interim_dir: Path | None = None,
    processed_dir: Path | None = None,
    preprocess_summary: dict[str, Any] | None = None,
    feature_build_config: LoadedFeatureBuildConfig | None = None,
    home_bank_id: int | None = None,
) -> dict[str, Any]:
    interim_dir = interim_dir or DATA_INTERIM
    processed_dir = processed_dir or DATA_PROCESSED
    processed_dir.mkdir(parents=True, exist_ok=True)

    if feature_build_config is None:
        cfg_path = default_feature_build_config_path()
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Default feature config not found at {cfg_path}. "
                "Pass feature_build_config or add config/feature_build.json."
            )
        feature_build_config = load_feature_build_config(cfg_path)

    if home_bank_id is not None and medium_bank_id != small_bank_id:
        logger.warning(
            "home_bank_id=%s ignored; using medium_bank_id=%s, small_bank_id=%s",
            home_bank_id,
            medium_bank_id,
            small_bank_id,
        )
    flags = feature_build_config.flags
    logger.info(
        "build_all_feature_tables: medium_bank_id=%s, small_bank_id=%s",
        medium_bank_id,
        small_bank_id,
    )
    output_paths = feature_parquet_paths(
        medium_bank_id=medium_bank_id,
        small_bank_id=small_bank_id,
        processed_dir=processed_dir,
    )

    salt = ensure_hash_salt(interim_dir)
    fx = load_or_create_fx(interim_dir)

    available_splits = _resolve_available_splits(
        path_medium, path_small, preprocess_summary
    )
    if not available_splits:
        raise FileNotFoundError(
            f"No home-bank Parquets found for feature build "
            f"(medium={path_medium}, small={path_small})"
        )

    med: pd.DataFrame | None = None
    sml: pd.DataFrame | None = None
    bank_filter_stats: dict[str, dict[str, int]] = {}
    if DATA_SPLIT_MEDIUM in available_splits:
        med = pd.read_parquet(path_medium)
        logger.info("loaded Medium Parquet: %s rows", len(med))
        med, stat_med = _filter_home_bank_rows(med, medium_bank_id, split_label="medium")
        bank_filter_stats["medium"] = stat_med
    if DATA_SPLIT_SMALL in available_splits:
        sml = pd.read_parquet(path_small)
        logger.info("loaded Small Parquet: %s rows", len(sml))
        sml, stat_sml = _filter_home_bank_rows(sml, small_bank_id, split_label="small")
        bank_filter_stats["small"] = stat_sml

    if med is not None:
        logger.info("enrich_transaction_frame: Medium …")
        med = enrich_transaction_frame(med, medium_bank_id, salt, fx)
    if sml is not None:
        logger.info("enrich_transaction_frame: Small …")
        sml = enrich_transaction_frame(sml, small_bank_id, salt, fx)

    fit_df = med if med is not None else sml
    assert fit_df is not None
    fit_bank_id = medium_bank_id if med is not None else small_bank_id

    if flags.corridor_risk_score:
        corridor_fit = fit_corridor_scores(fit_df, interim_dir)
    else:
        corridor_fit = {"weights": {}, "max_count": 1.0}
        interim_dir.mkdir(parents=True, exist_ok=True)
        (interim_dir / CORRIDOR_FIT_FILENAME).write_text(
            json.dumps(corridor_fit, indent=2), encoding="utf-8"
        )
        logger.info("corridor_risk_score disabled: wrote empty corridor fit and using score 0.0")
    if med is not None:
        med["corridor_risk_score"] = apply_corridor_scores(med, corridor_fit)
    if sml is not None:
        sml["corridor_risk_score"] = apply_corridor_scores(sml, corridor_fit)

    if med is not None:
        med = med.copy()
        med[EVENT_TIMESTAMP_COL] = pd.to_datetime(med[EVENT_TIMESTAMP_COL], utc=True)
    if sml is not None:
        sml = sml.copy()
        sml[EVENT_TIMESTAMP_COL] = pd.to_datetime(sml[EVENT_TIMESTAMP_COL], utc=True)

    if flags.weekly_internal_graph:
        logger.info(
            "fitting weekly internal graph features from %s …",
            DATA_SPLIT_MEDIUM if med is not None else DATA_SPLIT_SMALL,
        )
        graph_df = build_weekly_internal_graph_features(fit_df, fit_bank_id)
        logger.info("graph feature rows: %s", len(graph_df))
        for split_label, frame in ((DATA_SPLIT_MEDIUM, med), (DATA_SPLIT_SMALL, sml)):
            if frame is None:
                continue
            ts = frame[EVENT_TIMESTAMP_COL]
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            frame["_week"] = ts.dt.to_period("W-MON").astype(str)
            merged = frame.merge(
                graph_df,
                how="left",
                left_on=[ENTITY_ID_COL, "_week"],
                right_on=[ENTITY_ID_COL, "period"],
            )
            merged["graph_internal_degree"] = (
                pd.to_numeric(merged["graph_internal_degree"], errors="coerce")
                .fillna(0)
                .astype("Int64")
            )
            merged["graph_component_size"] = (
                pd.to_numeric(merged["graph_component_size"], errors="coerce")
                .fillna(1)
                .astype("Int64")
            )
            merged.drop(columns=["_week", "period"], inplace=True, errors="ignore")
            if split_label == DATA_SPLIT_MEDIUM:
                med = merged
            else:
                sml = merged
    else:
        logger.info("weekly_internal_graph disabled: using degree=0, component_size=1")
        for frame in (med, sml):
            if frame is None:
                continue
            frame["graph_internal_degree"] = pd.Series(0, index=frame.index, dtype="Int64")
            frame["graph_component_size"] = pd.Series(1, index=frame.index, dtype="Int64")

    for frame in (med, sml):
        if frame is None:
            continue
        frame["counterparty_bank_id"] = (
            pd.to_numeric(frame["counterparty_bank_id"], errors="coerce")
            .fillna(-1)
            .astype("int64")
        )

    logger.info("computing rolling / proxy features per split …")
    med_out = (
        _apply_rolling_and_proxy_features(med, bank_id=medium_bank_id, salt=salt, flags=flags)
        if med is not None
        else None
    )
    sml_out = (
        _apply_rolling_and_proxy_features(sml, bank_id=small_bank_id, salt=salt, flags=flags)
        if sml is not None
        else None
    )

    txn_cols = [
        TRANSACTION_ID_COL,
        EVENT_TIMESTAMP_COL,
        "data_split_source",
        ENTITY_ID_COL,
        "from_bank",
        "to_bank",
        "is_outbound",
        "counterparty_bank_id",
        "external_account_hash",
        "amount_usd",
        "amount_round_100",
        "corridor_risk_score",
        "velocity_24h_outbound",
        "dwell_sec_since_last_inbound",
        "fanout_unique_internal_7d",
        "fanout_unique_internal_30d",
        "graph_internal_degree",
        "graph_component_size",
        "hrj_country_flag",
        "pep_proxy_synthetic",
        "company_age_days_proxy",
        LABEL_COL,
    ]
    acc_txn_cols = [
        ENTITY_ID_COL,
        EVENT_TIMESTAMP_COL,
        "data_split_source",
        "velocity_24h_outbound",
        "dwell_sec_since_last_inbound",
        "fanout_unique_internal_7d",
        "fanout_unique_internal_30d",
        "company_age_days_proxy",
    ]
    acc_graph_cols = [
        ENTITY_ID_COL,
        EVENT_TIMESTAMP_COL,
        "graph_internal_degree",
        "graph_component_size",
    ]
    base_cols = [
        TRANSACTION_ID_COL,
        EVENT_TIMESTAMP_COL,
        ENTITY_ID_COL,
        "data_split_source",
        LABEL_COL,
    ]
    feat_cols = [c for c in txn_cols if c not in set(base_cols)]

    written_outputs: dict[str, Path] = {}
    if med_out is not None:
        txn_med = med_out[[c for c in txn_cols if c in med_out.columns]]
        logger.info(
            "writing %s (%s rows) …",
            output_paths["txn_level_medium"].name,
            len(txn_med),
        )
        txn_med.to_parquet(output_paths["txn_level_medium"], index=False)
        written_outputs["txn_level_medium"] = output_paths["txn_level_medium"]
        med_out[acc_txn_cols].copy().to_parquet(
            output_paths["account_txn_medium"], index=False
        )
        written_outputs["account_txn_medium"] = output_paths["account_txn_medium"]
        med_out[acc_graph_cols].copy().to_parquet(
            output_paths["account_graph_medium"], index=False
        )
        written_outputs["account_graph_medium"] = output_paths["account_graph_medium"]
        daily_med = (
            build_account_daily(med_out)
            if flags.account_daily
            else _empty_account_daily_frame()
        )
        daily_med.to_parquet(output_paths["account_daily_medium"], index=False)
        written_outputs["account_daily_medium"] = output_paths["account_daily_medium"]
        exp_med = med_out[base_cols + feat_cols].copy()
        logger.info(
            "writing experiment Parquet: %s (%s rows) …",
            output_paths["experiment_medium"].name,
            len(exp_med),
        )
        exp_med.to_parquet(output_paths["experiment_medium"], index=False)
        written_outputs["experiment_medium"] = output_paths["experiment_medium"]

    if sml_out is not None:
        txn_sml = sml_out[[c for c in txn_cols if c in sml_out.columns]]
        logger.info(
            "writing %s (%s rows) …",
            output_paths["txn_level_small"].name,
            len(txn_sml),
        )
        txn_sml.to_parquet(output_paths["txn_level_small"], index=False)
        written_outputs["txn_level_small"] = output_paths["txn_level_small"]
        sml_out[acc_txn_cols].copy().to_parquet(
            output_paths["account_txn_small"], index=False
        )
        written_outputs["account_txn_small"] = output_paths["account_txn_small"]
        sml_out[acc_graph_cols].copy().to_parquet(
            output_paths["account_graph_small"], index=False
        )
        written_outputs["account_graph_small"] = output_paths["account_graph_small"]
        daily_sml = (
            build_account_daily(sml_out)
            if flags.account_daily
            else _empty_account_daily_frame()
        )
        daily_sml.to_parquet(output_paths["account_daily_small"], index=False)
        written_outputs["account_daily_small"] = output_paths["account_daily_small"]
        exp_sml = sml_out[base_cols + feat_cols].copy()
        logger.info(
            "writing experiment Parquet: %s (%s rows) …",
            output_paths["experiment_small"].name,
            len(exp_sml),
        )
        exp_sml.to_parquet(output_paths["experiment_small"], index=False)
        written_outputs["experiment_small"] = output_paths["experiment_small"]

    if not flags.account_daily and (med_out is not None or sml_out is not None):
        logger.info("account_daily disabled: writing empty Parquets with schema preserved")

    manifest: dict[str, Any] = {
        "medium_bank_id": medium_bank_id,
        "small_bank_id": small_bank_id,
        "available_splits": sorted(available_splits),
        "git_sha": _git_sha(),
        "feature_config_signature": feature_build_config.signature,
        "feature_config": {
            "path": str(feature_build_config.source_path),
            "groups": feature_build_config.groups_meta,
            "bank_filter": bank_filter_stats,
        },
        "outputs": {k: str(v.resolve()) for k, v in written_outputs.items()},
        "output_filenames": {
            k: v.name for k, v in written_outputs.items()
        },
        "interim_artifacts": {
            "hash_salt_file": str((interim_dir / SALT_FILENAME).resolve()),
            "fx_rates": str((interim_dir / FX_FILENAME).resolve()),
            "corridor_fit": str((interim_dir / CORRIDOR_FIT_FILENAME).resolve()),
        },
    }
    if medium_bank_id == small_bank_id:
        manifest["home_bank_id"] = medium_bank_id
    if preprocess_summary is not None:
        manifest["preprocess_summary"] = preprocess_summary
    (interim_dir / MANIFEST_JSON).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
