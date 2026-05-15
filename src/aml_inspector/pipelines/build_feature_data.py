"""One-shot pipeline: home-bank preprocess → feature Parquets → optional ``feast apply``.

Run from repo root::

  python -m aml_inspector.pipelines.build_feature_data --bank-id 4 70 123
  python -m aml_inspector.pipelines.build_feature_data --medium-bank-id 70 --small-bank-id 42
  python -m aml_inspector.pipelines.build_feature_data --bank-id 123 --apply-feast
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aml_inspector.config import DATA_INTERIM, DATA_PROCESSED, FEAST_REPO
from aml_inspector.data.datasets import (
    MANIFEST_JSON,
    PARQUET_HOME_MEDIUM,
    PARQUET_HOME_SMALL,
    bank_scoped_output_subdir,
    feature_output_interim_dir,
    feature_output_processed_dir,
    resolve_feature_output_subdir,
    split_pair_scoped_output_subdir,
)
from aml_inspector.data.preprocess_home_bank import (
    DEFAULT_SUMMARY,
    MissingBankRowsError,
    run_preprocess_medium_small,
)
from aml_inspector.features.build_tables import build_all_feature_tables, feature_artifacts_present
from aml_inspector.features.feature_build_config import (
    LoadedFeatureBuildConfig,
    default_feature_build_config_path,
    load_feature_build_config,
)
from aml_inspector.features.materialize import feast_apply

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureBuildRunSpec:
    """One preprocess + feature-build target under scoped processed/interim dirs."""

    output_subdir: str | None
    bank_id: int | None = None
    medium_bank_id: int | None = None
    small_bank_id: int | None = None


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


def _bank_ids_from_spec(spec: FeatureBuildRunSpec) -> tuple[int, int, str] | None:
    """Return ``(medium_bank_id, small_bank_id, output_subdir)`` when CLI fixed them."""
    if spec.bank_id is not None:
        bid = int(spec.bank_id)
        subdir = spec.output_subdir or bank_scoped_output_subdir(bank_id=bid)
        return bid, bid, subdir
    if spec.medium_bank_id is not None and spec.small_bank_id is not None:
        med = int(spec.medium_bank_id)
        sml = int(spec.small_bank_id)
        subdir = spec.output_subdir or resolve_feature_output_subdir(
            medium_bank_id=med, small_bank_id=sml
        )
        return med, sml, subdir
    return None


def _feature_build_result_payload(
    *,
    output_subdir: str,
    medium_bank_id: int,
    small_bank_id: int,
    processed: Path,
    interim: Path,
    manifest_path: Path,
    summary: dict[str, Any] | None = None,
    reused_existing: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "output_subdir": output_subdir,
        "medium_bank_id": medium_bank_id,
        "small_bank_id": small_bank_id,
        "processed_dir": str(processed.resolve()),
        "interim_dir": str(interim.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    if reused_existing:
        result["reused_existing"] = True
    if medium_bank_id == small_bank_id:
        result["home_bank_id"] = medium_bank_id
    if summary is not None:
        if "available_splits" in summary:
            result["available_splits"] = summary["available_splits"]
        if summary.get("missing_splits"):
            result["missing_splits"] = summary["missing_splits"]
    return result


def _try_reuse_existing_feature_build(
    spec: FeatureBuildRunSpec,
    *,
    force_rebuild: bool,
    processed_root: Path,
    interim_root: Path,
    loaded_feature_cfg: LoadedFeatureBuildConfig,
) -> dict[str, Any] | None:
    """Skip preprocess + feature build when scoped outputs already exist."""
    if force_rebuild:
        return None
    resolved = _bank_ids_from_spec(spec)
    if resolved is None:
        return None
    medium_bank_id, small_bank_id, output_subdir = resolved
    processed = feature_output_processed_dir(
        output_subdir=output_subdir, processed_root=processed_root
    )
    interim = feature_output_interim_dir(
        output_subdir=output_subdir, interim_root=interim_root
    )
    existing_manifest = feature_artifacts_present(
        medium_bank_id=medium_bank_id,
        small_bank_id=small_bank_id,
        processed_dir=processed,
        interim_dir=interim,
        feature_config_signature=loaded_feature_cfg.signature,
    )
    if existing_manifest is None:
        return None
    logger.info(
        "Skipping %s (manifest + feature Parquets already present; "
        "use --force-rebuild to regenerate)",
        output_subdir,
    )
    summary_path = interim / DEFAULT_SUMMARY
    summary: dict[str, Any] | None = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = None
    manifest_path = interim / MANIFEST_JSON
    logger.info("Feature tables ready; manifest=%s", manifest_path.resolve())
    for key, path in sorted(existing_manifest.get("outputs", {}).items()):
        logger.info("  %s: %s", key, path)
    return _feature_build_result_payload(
        output_subdir=output_subdir,
        medium_bank_id=medium_bank_id,
        small_bank_id=small_bank_id,
        processed=processed,
        interim=interim,
        manifest_path=manifest_path,
        summary=summary,
        reused_existing=True,
    )


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


def plan_feature_build_runs(
    *,
    bank_ids: list[int] | None,
    medium_bank_id: int | None,
    small_bank_id: int | None,
) -> list[FeatureBuildRunSpec]:
    """Return one or more scoped build targets from CLI bank arguments."""
    has_split = medium_bank_id is not None or small_bank_id is not None
    if has_split and bank_ids:
        raise ValueError(
            "Cannot combine --bank-id with --medium-bank-id / --small-bank-id. "
            "Use split-pair mode or one or more --bank-id values, not both."
        )
    if has_split:
        if medium_bank_id is None or small_bank_id is None:
            raise ValueError(
                "Split-pair mode requires both --medium-bank-id and --small-bank-id"
            )
        return [
            FeatureBuildRunSpec(
                output_subdir=split_pair_scoped_output_subdir(
                    medium_bank_id=medium_bank_id,
                    small_bank_id=small_bank_id,
                ),
                medium_bank_id=medium_bank_id,
                small_bank_id=small_bank_id,
            )
        ]
    if bank_ids:
        return [
            FeatureBuildRunSpec(
                output_subdir=bank_scoped_output_subdir(bank_id=b),
                bank_id=b,
            )
            for b in bank_ids
        ]
    return [FeatureBuildRunSpec(output_subdir=None)]


def run_feature_build(
    spec: FeatureBuildRunSpec,
    *,
    raw_dir: Path | None,
    medium_file: str,
    small_file: str,
    skip_preprocess: bool,
    force_rebuild: bool,
    min_positive: int,
    min_negative: int,
    processed_root: Path,
    interim_root: Path,
    loaded_feature_cfg: LoadedFeatureBuildConfig,
) -> dict[str, Any]:
    """Preprocess (optional) and build features for one scoped output directory."""
    early = _try_reuse_existing_feature_build(
        spec,
        force_rebuild=force_rebuild,
        processed_root=processed_root,
        interim_root=interim_root,
        loaded_feature_cfg=loaded_feature_cfg,
    )
    if early is not None:
        return early

    summary: dict[str, Any]
    output_subdir = spec.output_subdir

    if not skip_preprocess:
        interim_for_preprocess = (
            feature_output_interim_dir(output_subdir=output_subdir, interim_root=interim_root)
            if output_subdir is not None
            else interim_root
        )
        processed_for_preprocess = (
            feature_output_processed_dir(
                output_subdir=output_subdir, processed_root=processed_root
            )
            if output_subdir is not None
            else processed_root
        )
        interim_for_preprocess.mkdir(parents=True, exist_ok=True)
        processed_for_preprocess.mkdir(parents=True, exist_ok=True)
        summary_path = interim_for_preprocess / DEFAULT_SUMMARY

        logger.info(
            "Preprocessing into %s (bank_id=%s, medium_bank_id=%s, small_bank_id=%s)",
            processed_for_preprocess.resolve(),
            spec.bank_id,
            spec.medium_bank_id,
            spec.small_bank_id,
        )
        try:
            summary = run_preprocess_medium_small(
                raw_dir=raw_dir,
                bank_id=spec.bank_id,
                medium_bank_id=spec.medium_bank_id,
                small_bank_id=spec.small_bank_id,
                medium_file=medium_file,
                small_file=small_file,
                min_positive=min_positive,
                min_negative=min_negative,
                output_medium=processed_for_preprocess / PARQUET_HOME_MEDIUM,
                output_small=processed_for_preprocess / PARQUET_HOME_SMALL,
                summary_json=summary_path,
            )
        except MissingBankRowsError as e:
            logger.warning("Skipping %s: %s", output_subdir or "feature build run", e)
            result: dict[str, Any] = {
                "status": "skipped",
                "skip_reason": str(e),
                "output_subdir": output_subdir,
                "bank_id": spec.bank_id,
                "medium_bank_id": spec.medium_bank_id,
                "small_bank_id": spec.small_bank_id,
                "missing_bank_id": e.bank_id,
                "missing_split": e.data_split_source,
                "missing_input_file": str(e.path),
                "processed_dir": str(processed_for_preprocess.resolve()),
                "interim_dir": str(interim_for_preprocess.resolve()),
            }
            if spec.bank_id is not None:
                result["home_bank_id"] = spec.bank_id
            return result
    else:
        if output_subdir is None:
            raise ValueError(
                "--skip-preprocess requires an explicit output scope "
                "(--bank-id or --medium-bank-id with --small-bank-id)"
            )
        interim_for_preprocess = feature_output_interim_dir(
            output_subdir=output_subdir, interim_root=interim_root
        )
        processed_for_preprocess = feature_output_processed_dir(
            output_subdir=output_subdir, processed_root=processed_root
        )
        summary_path = interim_for_preprocess / DEFAULT_SUMMARY
        med_p = processed_for_preprocess / PARQUET_HOME_MEDIUM
        sml_p = processed_for_preprocess / PARQUET_HOME_SMALL
        logger.info(
            "Skipping preprocess; reusing %s, %s and %s",
            med_p.resolve(),
            sml_p.resolve(),
            summary_path.resolve(),
        )
        if not med_p.is_file() and not sml_p.is_file():
            raise FileNotFoundError(
                f"--skip-preprocess requires at least one of {med_p} or {sml_p}"
            )
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    medium_bank_id = _resolve_medium_bank_id(
        summary=summary,
        bank_id=spec.bank_id,
        medium_bank_id=spec.medium_bank_id,
        train_eval_bank_id=None,
    )
    small_bank_id = _resolve_small_bank_id(
        summary=summary,
        bank_id=spec.bank_id,
        small_bank_id=spec.small_bank_id,
        test_bank_id=None,
    )

    if output_subdir is None:
        output_subdir = resolve_feature_output_subdir(
            medium_bank_id=medium_bank_id,
            small_bank_id=small_bank_id,
        )

    processed = feature_output_processed_dir(
        output_subdir=output_subdir, processed_root=processed_root
    )
    interim = feature_output_interim_dir(output_subdir=output_subdir, interim_root=interim_root)
    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    if not skip_preprocess and (
        processed_for_preprocess.resolve() != processed.resolve()
        or interim_for_preprocess.resolve() != interim.resolve()
    ):
        import shutil

        for name in (PARQUET_HOME_MEDIUM, PARQUET_HOME_SMALL):
            src = processed_for_preprocess / name
            dst = processed / name
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)
            elif (
                src.is_file()
                and dst.is_file()
                and src.resolve() != dst.resolve()
            ):
                shutil.copy2(src, dst)
        src_summary = interim_for_preprocess / DEFAULT_SUMMARY
        dst_summary = interim / DEFAULT_SUMMARY
        if src_summary.is_file() and src_summary.resolve() != dst_summary.resolve():
            shutil.copy2(src_summary, dst_summary)

    logger.info(
        "Scoped build %s: medium_bank_id=%s, small_bank_id=%s",
        output_subdir,
        medium_bank_id,
        small_bank_id,
    )

    med_p = processed / PARQUET_HOME_MEDIUM
    sml_p = processed / PARQUET_HOME_SMALL
    if not med_p.is_file() and not sml_p.is_file():
        raise FileNotFoundError(
            f"Feature build requires at least one home-bank Parquet under {processed}; "
            f"neither {med_p.name} nor {sml_p.name} exists"
        )

    existing_manifest = (
        None
        if force_rebuild
        else feature_artifacts_present(
            medium_bank_id=medium_bank_id,
            small_bank_id=small_bank_id,
            processed_dir=processed,
            interim_dir=interim,
            feature_config_signature=loaded_feature_cfg.signature,
        )
    )
    reused_existing = existing_manifest is not None
    if reused_existing:
        logger.info(
            "Skipping feature build for %s (use --force-rebuild to regenerate)",
            output_subdir,
        )
        manifest = existing_manifest
    else:
        manifest = build_all_feature_tables(
            medium_bank_id=medium_bank_id,
            small_bank_id=small_bank_id,
            path_medium=med_p,
            path_small=sml_p,
            interim_dir=interim,
            processed_dir=processed,
            preprocess_summary=summary,
            feature_build_config=loaded_feature_cfg,
        )

    manifest_path = interim / MANIFEST_JSON
    logger.info("Feature tables ready; manifest=%s", manifest_path.resolve())
    for key, path in sorted(manifest.get("outputs", {}).items()):
        logger.info("  %s: %s", key, path)

    return _feature_build_result_payload(
        output_subdir=output_subdir,
        medium_bank_id=medium_bank_id,
        small_bank_id=small_bank_id,
        processed=processed,
        interim=interim,
        manifest_path=manifest_path,
        summary=summary,
        reused_existing=reused_existing,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build split home-bank Parquets, AML feature tables, and experiment frames.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--bank-id",
        type=int,
        nargs="+",
        default=None,
        metavar="BANK_ID",
        help=(
            "One or more home bank ids (Medium and Small use the same id per run). "
            "Each id writes to data/processed/bank_<id>/ and data/interim/bank_<id>/"
        ),
    )
    parser.add_argument(
        "--medium-bank-id",
        "--train-eval-bank-id",
        type=int,
        default=None,
        dest="medium_bank_id",
        help="Home bank id for HI-Medium train/eval (split-pair mode; not with --bank-id)",
    )
    parser.add_argument(
        "--small-bank-id",
        "--test-bank-id",
        type=int,
        default=None,
        dest="small_bank_id",
        help="Home bank id for HI-Small test (split-pair mode; not with --bank-id)",
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory for HI-* CSVs")
    parser.add_argument("--medium-file", default="HI-Medium_Trans.csv")
    parser.add_argument("--small-file", default="HI-Small_Trans.csv")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Reuse existing scoped home_bank_transactions_{medium,small}.parquet",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed root (default: project data/processed); bank subdirs created under it",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=None,
        help="Interim root (default: project data/interim); bank subdirs created under it",
    )
    parser.add_argument(
        "--apply-feast",
        action="store_true",
        help="Run `feast apply` in feast_repo after all Parquet builds",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild feature Parquets even if manifest + outputs already exist",
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

    try:
        processed_root = args.processed_dir or DATA_PROCESSED
        interim_root = args.interim_dir or DATA_INTERIM
        processed_root.mkdir(parents=True, exist_ok=True)
        interim_root.mkdir(parents=True, exist_ok=True)

        feature_cfg_path = args.feature_config or default_feature_build_config_path()
        if not feature_cfg_path.is_file():
            print("ERROR: feature config not found:", feature_cfg_path.resolve(), file=sys.stderr)
            return 1

        runs = plan_feature_build_runs(
            bank_ids=args.bank_id,
            medium_bank_id=args.medium_bank_id,
            small_bank_id=args.small_bank_id,
        )

        logger.info(
            "Starting build_feature_data (%s run(s), processed_root=%s, interim_root=%s, "
            "skip_preprocess=%s, apply_feast=%s, force_rebuild=%s)",
            len(runs),
            processed_root.resolve(),
            interim_root.resolve(),
            args.skip_preprocess,
            args.apply_feast,
            args.force_rebuild,
        )

        loaded_feature_cfg = load_feature_build_config(feature_cfg_path)
        logger.info("Loaded feature config signature=%s", loaded_feature_cfg.signature)

        run_results: list[dict[str, Any]] = []
        for i, spec in enumerate(runs, start=1):
            logger.info("Run %s / %s …", i, len(runs))
            run_results.append(
                run_feature_build(
                    spec,
                    raw_dir=args.raw_dir,
                    medium_file=args.medium_file,
                    small_file=args.small_file,
                    skip_preprocess=args.skip_preprocess,
                    force_rebuild=args.force_rebuild,
                    min_positive=args.min_positive,
                    min_negative=args.min_negative,
                    processed_root=processed_root,
                    interim_root=interim_root,
                    loaded_feature_cfg=loaded_feature_cfg,
                )
            )

        if args.apply_feast:
            built_runs = [r for r in run_results if r.get("status") != "skipped"]
            if built_runs:
                logger.info("Running feast apply in %s …", FEAST_REPO.resolve())
                code = feast_apply(FEAST_REPO)
                if code != 0:
                    logger.error("feast apply exited with code %s", code)
                    return int(code)
                logger.info("feast apply completed successfully")
            else:
                logger.info("Skipping feast apply because all feature build runs were skipped")
    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        logger.error("Pipeline failed: %s", e)
        print("ERROR:", e, file=sys.stderr)
        return 1

    skipped_runs = [r for r in run_results if r.get("status") == "skipped"]
    final: dict[str, Any] = {
        "status": "ok",
        "runs": run_results,
        "skipped_runs": len(skipped_runs),
    }
    if len(run_results) == 1:
        final.update(run_results[0])
    logger.info("Pipeline finished successfully (%s run(s))", len(run_results))
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
