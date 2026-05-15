"""Holdout evaluation on combined HI-Medium/HI-Small with frozen Champion–Challenger artifacts."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from aml_inspector.config import DATA_INTERIM, DATA_PROCESSED, mlflow_tracking_uri
from aml_inspector.modeling.config import MLFLOW_EXPERIMENT_COMBINED
from aml_inspector.modeling.data import manifest_path
from aml_inspector.modeling.runner import run_holdout_evaluation

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
            "Evaluate frozen models on combined HI-Medium and HI-Small holdout (no retuning)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow training run id containing experiment_bundle artifacts",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="feature_build_manifest.json (default: data/interim/); manifest mode only",
    )
    parser.add_argument(
        "--testing-bank-ids",
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
        "--experiment",
        default=MLFLOW_EXPERIMENT_COMBINED,
        help="MLflow experiment for holdout run",
    )
    parser.add_argument("--run-name", default=None, help="MLflow holdout run name")
    parser.add_argument(
        "--no-bootstrap-ci",
        action="store_true",
        help="Skip bootstrap PR-AUC confidence interval on holdout",
    )
    args = parser.parse_args(argv)
    _ensure_cli_logging()

    os.environ.setdefault("MLFLOW_TRACKING_URI", mlflow_tracking_uri())

    if args.testing_bank_ids and args.manifest:
        print(
            "ERROR: use either --testing-bank-ids or --manifest, not both.",
            file=sys.stderr,
        )
        return 1

    manifest_file: Path | None = None
    if args.testing_bank_ids is None:
        manifest_file = args.manifest or manifest_path(DATA_INTERIM)
        if not manifest_file.is_file():
            print("ERROR: manifest not found:", manifest_file.resolve(), file=sys.stderr)
            return 1

    try:
        run_id = run_holdout_evaluation(
            train_run_id=args.run_id,
            manifest_file=manifest_file,
            testing_bank_ids=args.testing_bank_ids,
            processed_dir=args.processed_dir or DATA_PROCESSED,
            experiment_name=args.experiment,
            run_name=args.run_name,
            bootstrap_ci=not args.no_bootstrap_ci,
        )
    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        logger.error("Evaluation failed: %s", e)
        print("ERROR:", e, file=sys.stderr)
        return 1

    print(f"Holdout MLflow run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
