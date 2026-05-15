"""Feature matrix preparation from experiment entity frames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from aml_inspector.data.datasets import LABEL_COL
from aml_inspector.features.feature_build_config import (
    FEATURE_GROUP_KEYS,
    FeatureBuildFlags,
    LoadedFeatureBuildConfig,
    default_feature_build_config_path,
    load_feature_build_config,
)
from aml_inspector.modeling.data import NON_FEATURE_COLUMNS

# Model columns produced per enabled group in config/feature_build.json
FEATURE_GROUP_MODEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "base_transaction": (
        "is_outbound",
        "counterparty_bank_id",
        "amount_usd",
        "amount_round_100",
    ),
    "corridor_risk_score": ("corridor_risk_score",),
    "weekly_internal_graph": ("graph_internal_degree", "graph_component_size"),
    "rolling_account_activity": (
        "velocity_24h_outbound",
        "dwell_sec_since_last_inbound",
        "fanout_unique_internal_7d",
        "fanout_unique_internal_30d",
    ),
    "company_age_days_proxy": ("company_age_days_proxy",),
    "pep_proxy_synthetic": ("pep_proxy_synthetic",),
    "hrj_country_flag": ("hrj_country_flag",),
    "account_daily": ("daily_tx_count", "daily_amount_sum"),
}


def model_columns_for_flags(flags: FeatureBuildFlags) -> list[str]:
    """Return ordered model column names for enabled feature groups."""
    cols: list[str] = []
    for key in FEATURE_GROUP_KEYS:
        if not getattr(flags, key):
            continue
        for col in FEATURE_GROUP_MODEL_COLUMNS[key]:
            if col not in cols:
                cols.append(col)
    return cols


def select_feature_columns_from_config(
    df: pd.DataFrame,
    feature_build_config: LoadedFeatureBuildConfig,
) -> list[str]:
    """Select model inputs from enabled groups in feature_build.json."""
    wanted = model_columns_for_flags(feature_build_config.flags)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(
            "Experiment frame is missing columns required by feature_build.json: "
            f"{missing}. Enabled groups: "
            f"{[k for k in FEATURE_GROUP_KEYS if getattr(feature_build_config.flags, k)]}"
        )
    cols = [c for c in wanted if c in df.columns]
    if not cols:
        raise ValueError("No feature columns selected from feature_build.json (all groups disabled?)")
    return cols


def select_feature_columns(
    df: pd.DataFrame,
    *,
    feature_build_config: LoadedFeatureBuildConfig | None = None,
) -> list[str]:
    """Numeric/bool model inputs, or config-driven columns when ``feature_build_config`` is set."""
    if feature_build_config is not None:
        return select_feature_columns_from_config(df, feature_build_config)

    cols: list[str] = []
    for c in df.columns:
        if c in NON_FEATURE_COLUMNS:
            continue
        if c.endswith("_hash") or c.startswith("Unnamed"):
            continue
        dtype = df[c].dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            cols.append(c)
    if not cols:
        raise ValueError("No feature columns selected from experiment frame")
    return cols


def load_default_feature_build_config() -> LoadedFeatureBuildConfig:
    """Load project default ``config/feature_build.json``."""
    path = default_feature_build_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature config: {path}")
    return load_feature_build_config(path)


def _to_float_frame(X: pd.DataFrame) -> np.ndarray:
    out = X.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(np.float64)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.to_numpy(dtype=np.float64)


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Impute missing values on the selected feature columns."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("cast", FunctionTransformer(_to_float_frame, validate=False)),
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def extract_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_columns].copy()
    if LABEL_COL not in df.columns:
        raise KeyError(f"Missing label column {LABEL_COL!r}")
    y = df[LABEL_COL].astype(bool).to_numpy()
    return X, y


def scale_pos_weight(y_train: np.ndarray) -> float:
    """XGBoost imbalance weight: neg_count / pos_count on train."""
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    if pos <= 0:
        raise ValueError("Training set has zero positive labels; cannot compute scale_pos_weight")
    return float(neg / pos)


def fit_transform_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_other: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    train_arr = preprocessor.fit_transform(X_train)
    other_arr = preprocessor.transform(X_other) if X_other is not None else None
    return train_arr, other_arr
