"""Train Champion (XGBoost) + Challenger (IsolationForest) on combined HI-Medium/HI-Small; log to MLflow."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow

from aml_inspector.config import DATA_INTERIM, DATA_PROCESSED, mlflow_tracking_uri
from aml_inspector.features.feature_build_config import default_feature_build_config_path
from aml_inspector.modeling.config import (
    DEFAULT_C_FALSE_ALARM,
    DEFAULT_C_MISS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_REVIEW_BUDGET_FRACTION,
    DEFAULT_VAL_SIZE,
    MLFLOW_EXPERIMENT_COMBINED,
)
from aml_inspector.modeling.config import ExperimentConfig
from aml_inspector.modeling.data import manifest_path
from aml_inspector.modeling.runner import run_training

logger = logging.getLogger(__name__)


def _ensure_cli_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train Champion + Challenger on combined HI-Medium and HI-Small "
            "(80/20 stratified val)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="feature_build_manifest.json (default: data/interim/); manifest mode only",
    )
    parser.add_argument(
        "--training-bank-ids",
        type=int,
        nargs="+",
        default=None,
        metavar="BANK_ID",
        help=(
            "One or more home bank ids; load all available HI-Medium and HI-Small "
            "Parquets from data/processed/bank_<id>/ and combine (skips missing "
            "bank/split pairs). Not used with --manifest."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed data root (default: data/processed)",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=None,
        help="feature_build.json (default: config/feature_build.json when using --training-bank-ids)",
    )
    parser.add_argument(
        "--experiment",
        default=MLFLOW_EXPERIMENT_COMBINED,
        help="MLflow experiment name",
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--c-miss", type=float, default=DEFAULT_C_MISS)
    parser.add_argument("--c-false-alarm", type=float, default=DEFAULT_C_FALSE_ALARM)
    parser.add_argument(
        "--review-budget-fraction",
        type=float,
        default=DEFAULT_REVIEW_BUDGET_FRACTION,
        help="Top fraction for challenger threshold on validation",
    )
    args = parser.parse_args(argv)
    _ensure_cli_logging()

    os.environ.setdefault("MLFLOW_TRACKING_URI", mlflow_tracking_uri())

    if args.training_bank_ids and args.manifest:
        print(
            "ERROR: use either --training-bank-ids or --manifest, not both.",
            file=sys.stderr,
        )
        return 1

    manifest_file: Path | None = None
    if args.training_bank_ids is None:
        manifest_file = args.manifest or manifest_path(DATA_INTERIM)
        if not manifest_file.is_file():
            print("ERROR: manifest not found:", manifest_file.resolve(), file=sys.stderr)
            return 1

    feature_config_file = args.feature_config
    if args.training_bank_ids is not None and feature_config_file is None:
        feature_config_file = default_feature_build_config_path()
        if not feature_config_file.is_file():
            print(
                "ERROR: feature config not found:",
                feature_config_file.resolve(),
                file=sys.stderr,
            )
            return 1

    cfg = ExperimentConfig(
        random_state=args.random_state,
        val_size=args.val_size,
        c_miss=args.c_miss,
        c_false_alarm=args.c_false_alarm,
        review_budget_fraction=args.review_budget_fraction,
    )

    try:
        run_id = run_training(
            manifest_file=manifest_file,
            training_bank_ids=args.training_bank_ids,
            processed_dir=args.processed_dir or DATA_PROCESSED,
            feature_config_file=feature_config_file,
            experiment_name=args.experiment,
            config=cfg,
            run_name=args.run_name,
        )
    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        logger.error("Training failed: %s", e)
        print("ERROR:", e, file=sys.stderr)
        return 1

    print(f"MLflow run_id={run_id}")
    print("Evaluate holdout: python -m aml_inspector.modeling.evaluate --run-id", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
