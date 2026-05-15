"""Run Feast CLI workflows from the project (apply, optional historical fetch).

Run from repo root after `cd feast_repo` pattern:
  feast apply
  feast materialize-incremental ...  # when you add materialization jobs
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from aml_inspector.config import FEAST_REPO
from aml_inspector.data.datasets import (
    DATASET_TOKEN_HI_MEDIUM,
    DATASET_TOKEN_HI_SMALL,
)


def feast_apply(
    repo: Path | None = None,
    *,
    bank_id: int | None = None,
    dataset_token: str | None = None,
    output_subdir: str | None = None,
) -> int:
    """Execute ``feast apply`` in the Feast repo directory.

    When ``bank_id`` and ``dataset_token`` are set, exports ``AML_FEATURE_*`` env vars
    so ``feast_repo/data_sources.py`` resolves scoped Parquet paths under
    ``data/processed/<output_subdir>/``.
    """
    root = repo or FEAST_REPO
    env = os.environ.copy()
    if bank_id is not None and dataset_token is not None:
        env["AML_FEATURE_BANK_ID"] = str(int(bank_id))
        env["AML_FEATURE_DATASET"] = dataset_token.strip().upper()
        if output_subdir:
            env["AML_FEATURE_OUTPUT_SUBDIR"] = output_subdir
        else:
            env.pop("AML_FEATURE_OUTPUT_SUBDIR", None)
    return subprocess.call(["feast", "apply"], cwd=str(root), env=env)


def feast_apply_for_feature_build_run(run: dict[str, object]) -> int:
    """Run ``feast apply`` using the first available split from a pipeline run result."""
    output_subdir = str(run["output_subdir"])
    medium_bank_id = int(run["medium_bank_id"])
    small_bank_id = int(run["small_bank_id"])
    available = {str(s) for s in (run.get("available_splits") or ("medium", "small"))}
    missing = {str(s) for s in (run.get("missing_splits") or ())}

    if "medium" in available and "medium" not in missing:
        return feast_apply(
            bank_id=medium_bank_id,
            dataset_token=DATASET_TOKEN_HI_MEDIUM,
            output_subdir=output_subdir,
        )
    if "small" in available and "small" not in missing:
        return feast_apply(
            bank_id=small_bank_id,
            dataset_token=DATASET_TOKEN_HI_SMALL,
            output_subdir=output_subdir,
        )
    raise ValueError(
        f"No feature Parquet split available for feast apply (run={output_subdir!r})"
    )
