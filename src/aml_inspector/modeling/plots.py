"""Optional metric plots logged as MLflow artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    *,
    title: str = "Precision-Recall",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true.astype(bool), y_score, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    *,
    title: str = "ROC",
) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true.astype(bool), y_score, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def log_metric_plots(
    y_true: np.ndarray,
    champion_prob: np.ndarray,
    *,
    prefix: str,
) -> None:
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pr_path = root / f"{prefix}_pr_curve.png"
        roc_path = root / f"{prefix}_roc_curve.png"
        save_pr_curve(y_true, champion_prob, pr_path, title=f"{prefix} PR")
        save_roc_curve(y_true, champion_prob, roc_path, title=f"{prefix} ROC")
        mlflow.log_artifact(str(pr_path), artifact_path=f"plots/{prefix}")
        if roc_path.is_file():
            mlflow.log_artifact(str(roc_path), artifact_path=f"plots/{prefix}")
