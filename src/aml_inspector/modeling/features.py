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
from aml_inspector.modeling.data import NON_FEATURE_COLUMNS


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric/bool model inputs; exclude ids, label, and hash columns."""
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
