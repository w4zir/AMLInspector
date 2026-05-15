"""Save and load experiment bundles via MLflow and local paths."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow

from aml_inspector.modeling.config import ExperimentBundle, ExperimentConfig, FrozenPolicy

BUNDLE_JSON = "experiment_bundle.json"
PREPROCESSOR_NAME = "preprocessor.joblib"
CHAMPION_NAME = "champion_xgb.joblib"
CHALLENGER_NAME = "challenger_iforest.joblib"


def save_bundle_artifacts(
    bundle: ExperimentBundle,
    *,
    preprocessor: Any,
    champion_model: Any,
    challenger_model: Any,
    artifact_path: str = "experiment_bundle",
) -> None:
    """Log bundle JSON and fitted models to the active MLflow run."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / BUNDLE_JSON).write_text(
            json.dumps(bundle.to_dict(), indent=2),
            encoding="utf-8",
        )
        joblib.dump(preprocessor, root / PREPROCESSOR_NAME)
        joblib.dump(champion_model, root / CHAMPION_NAME)
        joblib.dump(challenger_model, root / CHALLENGER_NAME)
        mlflow.log_artifacts(str(root), artifact_path=artifact_path)


def load_bundle_from_run(
    run_id: str,
    *,
    artifact_path: str = "experiment_bundle",
) -> tuple[ExperimentBundle, Any, Any, Any]:
    """Download bundle from an MLflow run and deserialize models."""
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp)
        client.download_artifacts(run_id, artifact_path, dst_path=str(local))
        root = local / artifact_path if (local / artifact_path).is_dir() else local
        bundle_dict = json.loads((root / BUNDLE_JSON).read_text(encoding="utf-8"))
        bundle = _bundle_from_dict(bundle_dict)
        preprocessor = joblib.load(root / PREPROCESSOR_NAME)
        champion = joblib.load(root / CHAMPION_NAME)
        challenger = joblib.load(root / CHALLENGER_NAME)
        return bundle, preprocessor, champion, challenger


def _bundle_from_dict(d: dict[str, Any]) -> ExperimentBundle:
    from aml_inspector.modeling.config import SplitCounts

    policy_d = d["frozen_policy"]
    cfg_d = d["experiment_config"]
    split_raw = d.get("split_counts", {})
    split_counts = {
        k: SplitCounts(**v) if isinstance(v, dict) else v for k, v in split_raw.items()
    }
    return ExperimentBundle(
        feature_columns=list(d["feature_columns"]),
        frozen_policy=FrozenPolicy(**policy_d),
        experiment_config=ExperimentConfig(**cfg_d),
        scale_pos_weight=float(d["scale_pos_weight"]),
        medium_bank_id=int(d["medium_bank_id"]),
        small_bank_id=int(d["small_bank_id"]),
        feature_manifest_id=str(d["feature_manifest_id"]),
        git_sha=str(d["git_sha"]),
        split_counts=split_counts,
        champion_params=dict(d.get("champion_params", {})),
        challenger_params=dict(d.get("challenger_params", {})),
    )


def log_run_tags(
    *,
    split_policy: str,
    data_revision: str,
    home_bank_id: int | str,
    model_role: str,
    medium_bank_id: int | None = None,
    small_bank_id: int | None = None,
) -> None:
    mlflow.set_tag("split_policy", split_policy)
    mlflow.set_tag("data_revision", data_revision)
    mlflow.set_tag("home_bank_id", str(home_bank_id))
    mlflow.set_tag("model_role", model_role)
    if medium_bank_id is not None:
        mlflow.set_tag("medium_bank_id", str(medium_bank_id))
    if small_bank_id is not None:
        mlflow.set_tag("small_bank_id", str(small_bank_id))
