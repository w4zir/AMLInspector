"""Stratified train/validation splits on HI-Medium only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aml_inspector.data.datasets import LABEL_COL
from aml_inspector.modeling.config import SplitCounts


def stratified_train_val_split(
    df: pd.DataFrame,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split on label within Medium frame."""
    if LABEL_COL not in df.columns:
        raise KeyError(f"Missing {LABEL_COL!r} for stratified split")
    y = df[LABEL_COL].astype(bool)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 1 or n_neg < 1:
        raise ValueError(
            f"Need both classes for stratified split; positives={n_pos}, negatives={n_neg}"
        )
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def split_counts(df: pd.DataFrame) -> SplitCounts:
    y = df[LABEL_COL].astype(bool)
    n_pos = int(y.sum())
    n = len(df)
    return SplitCounts(
        n_rows=n,
        n_positives=n_pos,
        positive_rate=float(n_pos / n) if n else 0.0,
    )
