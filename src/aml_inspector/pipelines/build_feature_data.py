"""One-shot pipeline: home-bank preprocess → feature Parquets → optional ``feast apply``.

Run from repo root::

  python -m aml_inspector.pipelines.build_feature_data --medium-bank-id 70 --small-bank-id 42
  python -m aml_inspector.pipelines.build_feature_data --bank-id 123 --apply-feast
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from aml_inspector.config import DATA_INTERIM, DATA_PROCESSED, FEAST_REPO
from aml_inspector.data.datasets import MANIFEST_JSON, PARQUET_HOME_MEDIUM, PARQUET_HOME_SMALL
from aml_inspector.data.preprocess_home_bank import DEFAULT_SUMMARY, run_preprocess_medium_small
from aml_inspector.features.build_tables import build_all_feature_tables, feature_artifacts_present
from aml_inspector.features.feature_build_config import (
    default_feature_build_config_path,
    load_feature_build_config,
)
from aml_inspector.features.materialize import feast_apply

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


def _resolve_medium_bank_id(
    *,
    summary: dict[str, Any],
    bank_id: int | None,
    medium_bank_id: int | None,
    train_eval_bank_id: int | None,
) -> int:
    if medium_bank_id is not None:
        return int(medium_bank_id)
    if train_eval_bank_id is not None:
        return int(train_eval_bank_id)
    if bank_id is not None:
        return int(bank_id)
    if "medium_bank_id" in summary:
        return int(summary["medium_bank_id"])
    if "home_bank_id" in summary:
        return int(summary["home_bank_id"])
    raise KeyError("summary missing medium_bank_id / home_bank_id")


def _resolve_small_bank_id(
    *,
    summary: dict[str, Any],
    bank_id: int | None,
    small_bank_id: int | None,
    test_bank_id: int | None,
) -> int:
    if small_bank_id is not None:
        return int(small_bank_id)
    if test_bank_id is not None:
        return int(test_bank_id)
    if bank_id is not None:
        return int(bank_id)
    if "small_bank_id" in summary:
        return int(summary["small_bank_id"])
    if "home_bank_id" in summary:
        return int(summary["home_bank_id"])
    raise KeyError("summary missing small_bank_id / home_bank_id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build split home-bank Parquets, AML feature tables, and experiment frames.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--bank-id",
        type=int,
        default=None,
        help="Shared home bank id for Medium and Small when split-specific ids are omitted",
    )
    parser.add_argument(
        "--medium-bank-id",
        "--train-eval-bank-id",
        type=int,
        default=None,
        dest="medium_bank_id",
        help="Home bank id for HI-Medium train/eval (overrides --bank-id for Medium)",
    )
    parser.add_argument(
        "--small-bank-id",
        "--test-bank-id",
        type=int,
        default=None,
        dest="small_bank_id",
        help="Home bank id for HI-Small test (overrides --bank-id for Small)",
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory for HI-* CSVs")
    parser.add_argument("--medium-file", default="HI-Medium_Trans.csv")
    parser.add_argument("--small-file", default="HI-Small_Trans.csv")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Reuse existing home_bank_transactions_{medium,small}.parquet and interim summary",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Output directory (default: project data/processed)",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=None,
        help="Interim directory (default: project data/interim)",
    )
    parser.add_argument(
        "--apply-feast",
        action="store_true",
        help="Run `feast apply` in feast_repo after Parquet build",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild feature Parquets even if manifest + bank/dataset outputs already exist",
    )
    parser.add_argument("--min-positive", type=int, default=1)
    parser.add_argument("--min-negative", type=int, default=1)
    parser.add_argument(
        "--feature-config",
        "--feature",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON feature toggles (default: config/feature_build.json under project root)",
    )
    args = parser.parse_args(argv)
    _ensure_cli_logging()

    medium_bank_id: int
    small_bank_id: int
    summary: dict[str, Any]

    try:
        processed = args.processed_dir or DATA_PROCESSED
        interim = args.interim_dir or DATA_INTERIM
        processed.mkdir(parents=True, exist_ok=True)
        interim.mkdir(parents=True, exist_ok=True)

        summary_path = interim / DEFAULT_SUMMARY

        feature_cfg_path = args.feature_config or default_feature_build_config_path()
        if not feature_cfg_path.is_file():
            print("ERROR: feature config not found:", feature_cfg_path.resolve(), file=sys.stderr)
            return 1

        logger.info(
            "Starting build_feature_data (processed=%s, interim=%s, skip_preprocess=%s, "
            "apply_feast=%s, force_rebuild=%s, feature_config=%s)",
            processed.resolve(),
            interim.resolve(),
            args.skip_preprocess,
            args.apply_feast,
            args.force_rebuild,
            feature_cfg_path.resolve(),
        )

        loaded_feature_cfg = load_feature_build_config(feature_cfg_path)
        logger.info("Loaded feature config signature=%s", loaded_feature_cfg.signature)
        if not args.skip_preprocess:
            logger.info(
                "Preprocessing (raw_dir=%s, bank_id=%s, medium_bank_id=%s, small_bank_id=%s)",
                args.raw_dir,
                args.bank_id,
                args.medium_bank_id,
                args.small_bank_id,
            )
            summary = run_preprocess_medium_small(
                raw_dir=args.raw_dir,
                bank_id=args.bank_id,
                medium_bank_id=args.medium_bank_id,
                small_bank_id=args.small_bank_id,
                medium_file=args.medium_file,
                small_file=args.small_file,
                min_positive=args.min_positive,
                min_negative=args.min_negative,
                output_medium=processed / PARQUET_HOME_MEDIUM,
                output_small=processed / PARQUET_HOME_SMALL,
                summary_json=summary_path,
            )
        else:
            med_p = processed / PARQUET_HOME_MEDIUM
            sml_p = processed / PARQUET_HOME_SMALL
            logger.info(
                "Skipping preprocess; reusing Parquets %s, %s and summary %s",
                med_p.resolve(),
                sml_p.resolve(),
                summary_path.resolve(),
            )
            if not med_p.is_file() or not sml_p.is_file():
                print(
                    "ERROR: --skip-preprocess requires existing",
                    med_p,
                    "and",
                    sml_p,
                    file=sys.stderr,
                )
                return 1
            if not summary_path.is_file():
                print("ERROR: missing", summary_path, file=sys.stderr)
                return 1
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        medium_bank_id = _resolve_medium_bank_id(
            summary=summary,
            bank_id=args.bank_id,
            medium_bank_id=args.medium_bank_id,
            train_eval_bank_id=None,
        )
        small_bank_id = _resolve_small_bank_id(
            summary=summary,
            bank_id=args.bank_id,
            small_bank_id=args.small_bank_id,
            test_bank_id=None,
        )
        logger.info(
            "Using medium_bank_id=%s, small_bank_id=%s",
            medium_bank_id,
            small_bank_id,
        )

        existing_manifest = (
            None
            if args.force_rebuild
            else feature_artifacts_present(
                medium_bank_id=medium_bank_id,
                small_bank_id=small_bank_id,
                processed_dir=processed,
                interim_dir=interim,
                feature_config_signature=loaded_feature_cfg.signature,
            )
        )
        if existing_manifest is not None:
            logger.info(
                "Skipping feature Parquet build: outputs exist for medium_bank_id=%s, "
                "small_bank_id=%s (use --force-rebuild to regenerate)",
                medium_bank_id,
                small_bank_id,
            )
            manifest = existing_manifest
        else:
            logger.info(
                "Building feature Parquets (medium_bank_id=%s, small_bank_id=%s) …",
                medium_bank_id,
                small_bank_id,
            )
            manifest = build_all_feature_tables(
                medium_bank_id=medium_bank_id,
                small_bank_id=small_bank_id,
                path_medium=processed / PARQUET_HOME_MEDIUM,
                path_small=processed / PARQUET_HOME_SMALL,
                interim_dir=interim,
                processed_dir=processed,
                preprocess_summary=summary,
                feature_build_config=loaded_feature_cfg,
            )
        logger.info(
            "Feature tables ready; manifest=%s",
            (interim / MANIFEST_JSON).resolve(),
        )
        for key, path in sorted(manifest.get("outputs", {}).items()):
            logger.info("  %s: %s", key, path)

        if args.apply_feast:
            logger.info("Running feast apply in %s …", FEAST_REPO.resolve())
            code = feast_apply(FEAST_REPO)
            if code != 0:
                logger.error("feast apply exited with code %s", code)
                return int(code)
            logger.info("feast apply completed successfully")
    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        logger.error("Pipeline failed: %s", e)
        print("ERROR:", e, file=sys.stderr)
        return 1

    result: dict[str, Any] = {
        "status": "ok",
        "medium_bank_id": medium_bank_id,
        "small_bank_id": small_bank_id,
        "manifest": str((interim / MANIFEST_JSON).resolve()),
    }
    if medium_bank_id == small_bank_id:
        result["home_bank_id"] = medium_bank_id
    logger.info("Pipeline finished successfully")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
