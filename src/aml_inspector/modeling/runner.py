"""Training and holdout evaluation orchestration."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from aml_inspector.modeling.artifacts import (
    load_bundle_from_run,
    log_run_tags,
    save_bundle_artifacts,
)
from aml_inspector.modeling.config import (
    ExperimentBundle,
    ExperimentConfig,
    FrozenPolicy,
    MLFLOW_EXPERIMENT_COMBINED,
    SPLIT_POLICY_TAG,
)
from aml_inspector.modeling.data import (
    load_medium_small_frames,
    manifest_feature_id,
)
from aml_inspector.modeling.features import (
    build_preprocessor,
    extract_xy,
    fit_transform_preprocessor,
    scale_pos_weight,
    select_feature_columns,
)
from aml_inspector.modeling.metrics import (
    binary_metrics,
    bootstrap_pr_auc_ci,
    prefix_metrics,
    precision_at_fraction,
    recall_at_budget,
)
from aml_inspector.modeling.plots import log_metric_plots
from aml_inspector.modeling.policy import combined_policy_metrics
from aml_inspector.modeling.splits import split_counts, stratified_train_val_split
from aml_inspector.modeling.thresholds import (
    champion_soft_threshold,
    select_challenger_threshold,
    select_champion_threshold,
)

logger = logging.getLogger(__name__)

DEFAULT_CHAMPION_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 20,
}

DEFAULT_CHALLENGER_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": 42,
}


def _challenger_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negate sklearn decision_function)."""
    return (-model.decision_function(X)).astype(np.float64)


def run_training(
    *,
    manifest_file: Path | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_COMBINED,
    config: ExperimentConfig | None = None,
    champion_params: dict[str, Any] | None = None,
    challenger_params: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> str:
    """Train Champion + Challenger on Medium train/val; return MLflow run_id."""
    cfg = config or ExperimentConfig()
    c_params = {**DEFAULT_CHAMPION_PARAMS, **(champion_params or {})}
    ch_params = {**DEFAULT_CHALLENGER_PARAMS, **(challenger_params or {})}
    ch_params.setdefault("random_state", cfg.random_state)

    medium_df, _small_df, manifest = load_medium_small_frames(manifest_file=manifest_file)
    train_df, val_df = stratified_train_val_split(
        medium_df,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    feature_columns = select_feature_columns(medium_df)
    X_train, y_train = extract_xy(train_df, feature_columns)
    X_val, y_val = extract_xy(val_df, feature_columns)

    preprocessor = build_preprocessor(feature_columns)
    X_train_t, X_val_t = fit_transform_preprocessor(preprocessor, X_train, X_val)

    spw = scale_pos_weight(y_train)
    clf = XGBClassifier(
        max_depth=c_params.get("max_depth", 6),
        learning_rate=c_params.get("learning_rate", 0.1),
        n_estimators=c_params.get("n_estimators", 200),
        subsample=c_params.get("subsample", 0.8),
        colsample_bytree=c_params.get("colsample_bytree", 0.8),
        min_child_weight=c_params.get("min_child_weight", 1),
        scale_pos_weight=spw,
        random_state=cfg.random_state,
        eval_metric=c_params.get("eval_metric", "aucpr"),
        early_stopping_rounds=c_params.get("early_stopping_rounds", 20),
    )
    clf.fit(
        X_train_t,
        y_train.astype(int),
        eval_set=[(X_val_t, y_val.astype(int))],
        verbose=False,
    )
    val_prob = clf.predict_proba(X_val_t)[:, 1]

    iforest = IsolationForest(
        n_estimators=ch_params.get("n_estimators", 200),
        max_samples=ch_params.get("max_samples", "auto"),
        contamination=ch_params.get("contamination", "auto"),
        random_state=ch_params.get("random_state", cfg.random_state),
    )
    iforest.fit(X_train_t)
    val_chall = _challenger_scores(iforest, X_val_t)

    t_champ, cost_info = select_champion_threshold(
        y_val,
        val_prob,
        c_miss=cfg.c_miss,
        c_false_alarm=cfg.c_false_alarm,
    )
    t_chall = select_challenger_threshold(val_chall, budget_fraction=cfg.review_budget_fraction)
    t_soft = champion_soft_threshold(t_champ, cfg.champion_soft_threshold_margin)
    policy = FrozenPolicy(
        t_champ=t_champ,
        t_chall=t_chall,
        t_champ_soft=t_soft,
        policy_version=cfg.policy_version,
    )

    medium_id = int(manifest["medium_bank_id"])
    small_id = int(manifest["small_bank_id"])
    feature_id = manifest_feature_id(manifest)
    git_sha = str(manifest.get("git_sha", "unknown"))

    bundle = ExperimentBundle(
        feature_columns=feature_columns,
        frozen_policy=policy,
        experiment_config=cfg,
        scale_pos_weight=spw,
        medium_bank_id=medium_id,
        small_bank_id=small_id,
        feature_manifest_id=feature_id,
        git_sha=git_sha,
        split_counts={
            "train": split_counts(train_df),
            "validation": split_counts(val_df),
            "medium_full": split_counts(medium_df),
        },
        champion_params={k: v for k, v in c_params.items() if k != "early_stopping_rounds"},
        challenger_params=ch_params,
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        log_run_tags(
            split_policy=SPLIT_POLICY_TAG,
            data_revision=feature_id,
            home_bank_id=medium_id,
            model_role="ensemble_policy",
            medium_bank_id=medium_id,
            small_bank_id=small_id,
        )
        mlflow.log_params(
            {
                "split_policy": SPLIT_POLICY_TAG,
                "scale_pos_weight": spw,
                "t_champ": t_champ,
                "t_chall": t_chall,
                "t_champ_soft": t_soft,
                "c_miss": cfg.c_miss,
                "c_false_alarm": cfg.c_false_alarm,
                "review_budget_fraction": cfg.review_budget_fraction,
                "feature_manifest_id": feature_id,
                "n_features": len(feature_columns),
                **{f"champion_{k}": v for k, v in bundle.champion_params.items()},
                **{f"challenger_{k}": v for k, v in bundle.challenger_params.items()},
            }
        )
        for split_name, sc in bundle.split_counts.items():
            mlflow.log_metric(f"{split_name}_rows", sc.n_rows)
            mlflow.log_metric(f"{split_name}_positive_rate", sc.positive_rate)

        val_champ_metrics = binary_metrics(y_val, val_prob, threshold=t_champ)
        mlflow.log_metrics(prefix_metrics(val_champ_metrics, "val_champion"))
        mlflow.log_metrics(cost_info)

        combined = combined_policy_metrics(y_val, val_prob, val_chall, policy)
        mlflow.log_metrics(prefix_metrics(combined, "val_combined"))

        mlflow.log_metric(
            "val_challenger_precision_at_budget",
            precision_at_fraction(y_val, val_chall, cfg.review_budget_fraction),
        )
        mlflow.log_metric(
            "val_challenger_recall_at_budget",
            recall_at_budget(y_val, val_chall, cfg.review_budget_fraction),
        )

        log_metric_plots(y_val, val_prob, prefix="validation")

        save_bundle_artifacts(
            bundle,
            preprocessor=preprocessor,
            champion_model=clf,
            challenger_model=iforest,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "experiment_config.json"
            cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
            mlflow.log_artifact(str(cfg_path), artifact_path="config")

        run_id = run.info.run_id
        logger.info("Training complete run_id=%s", run_id)
        return run_id


def run_holdout_evaluation(
    *,
    train_run_id: str,
    manifest_file: Path | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_COMBINED,
    run_name: str | None = None,
    bootstrap_ci: bool = True,
) -> str:
    """Score HI-Small holdout with frozen models; return MLflow run_id."""
    bundle, preprocessor, champion, challenger = load_bundle_from_run(train_run_id)
    _medium_df, small_df, manifest = load_medium_small_frames(manifest_file=manifest_file)

    feature_columns = bundle.feature_columns
    X_test, y_test = extract_xy(small_df, feature_columns)
    X_test_t = preprocessor.transform(X_test)

    test_prob = champion.predict_proba(X_test_t)[:, 1]
    test_chall = _challenger_scores(challenger, X_test_t)
    policy = bundle.frozen_policy

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name or f"holdout_{train_run_id[:8]}") as run:
        mlflow.set_tag("parent_run_id", train_run_id)
        log_run_tags(
            split_policy=SPLIT_POLICY_TAG,
            data_revision=bundle.feature_manifest_id,
            home_bank_id=bundle.small_bank_id,
            model_role="holdout_test",
            medium_bank_id=bundle.medium_bank_id,
            small_bank_id=bundle.small_bank_id,
        )

        test_champ = binary_metrics(y_test, test_prob, threshold=policy.t_champ)
        mlflow.log_metrics(prefix_metrics(test_champ, "test_champion"))

        combined = combined_policy_metrics(y_test, test_prob, test_chall, policy)
        mlflow.log_metrics(prefix_metrics(combined, "test_combined"))

        if bootstrap_ci:
            ci = bootstrap_pr_auc_ci(
                y_test,
                test_prob,
                random_state=bundle.experiment_config.random_state,
            )
            mlflow.log_metrics({f"test_champion_{k}": v for k, v in ci.items()})

        sc = split_counts(small_df)
        mlflow.log_metric("test_rows", sc.n_rows)
        mlflow.log_metric("test_positive_rate", sc.positive_rate)

        log_metric_plots(y_test, test_prob, prefix="test")

        with tempfile.TemporaryDirectory() as tmp:
            results = {
                "train_run_id": train_run_id,
                "holdout_split": "small",
                "frozen_policy": policy.to_dict(),
                "test_champion_metrics": test_champ,
                "test_combined_metrics": combined,
                "split_counts": sc.to_dict(),
            }
            out = Path(tmp) / "holdout_results.json"
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(out), artifact_path="results")

        run_id = run.info.run_id
        logger.info("Holdout evaluation complete run_id=%s", run_id)
        return run_id
