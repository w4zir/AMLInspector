"""Run Feast CLI workflows from the project (apply, optional historical fetch).

Run from repo root after `cd feast_repo` pattern:
  feast apply
  feast materialize-incremental ...  # when you add materialization jobs
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from aml_inspector.config import FEAST_REPO


def feast_apply(repo: Path | None = None) -> int:
    """Execute `feast apply` in the Feast repo directory."""
    root = repo or FEAST_REPO
    return subprocess.call(["feast", "apply"], cwd=str(root))
