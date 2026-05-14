"""Train a baseline model and log to MLflow."""

from __future__ import annotations

import argparse
import os

import mlflow
from sklearn.dummy import DummyClassifier

from aml_inspector.config import mlflow_tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Train placeholder model (MLflow smoke test).")
    parser.add_argument("--experiment", default="aml_baseline", help="MLflow experiment name")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", mlflow_tracking_uri())
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        model = DummyClassifier(strategy="most_frequent")
        model.fit([[0], [1]], [0, 0])
        mlflow.log_param("model", "DummyClassifier")
        mlflow.log_metric("train_rows", 2)
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
