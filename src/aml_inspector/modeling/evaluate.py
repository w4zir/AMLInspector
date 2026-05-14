"""Evaluate a run or a local model artifact."""

from __future__ import annotations

import os

import mlflow

from aml_inspector.config import mlflow_tracking_uri


def main() -> None:
    os.environ.setdefault("MLFLOW_TRACKING_URI", mlflow_tracking_uri())

    with mlflow.start_run():
        mlflow.log_metric("val_auc", 0.5)


if __name__ == "__main__":
    main()
