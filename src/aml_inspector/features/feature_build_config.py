"""Load and validate ``feature_build.json`` for :func:`~aml_inspector.features.build_tables.build_all_feature_tables`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aml_inspector.config import PROJECT_ROOT

DEFAULT_FEATURE_BUILD_CONFIG_PATH = PROJECT_ROOT / "config" / "feature_build.json"

FEATURE_GROUP_KEYS = (
    "base_transaction",
    "corridor_risk_score",
    "weekly_internal_graph",
    "rolling_account_activity",
    "company_age_days_proxy",
    "pep_proxy_synthetic",
    "hrj_country_flag",
    "account_daily",
)


@dataclass(frozen=True)
class FeatureBuildFlags:
    """Resolved on/off flags per feature group."""

    base_transaction: bool
    corridor_risk_score: bool
    weekly_internal_graph: bool
    rolling_account_activity: bool
    company_age_days_proxy: bool
    pep_proxy_synthetic: bool
    hrj_country_flag: bool
    account_daily: bool


@dataclass(frozen=True)
class LoadedFeatureBuildConfig:
    """Config file payload plus a stable signature for manifest / cache skip checks."""

    flags: FeatureBuildFlags
    groups_meta: dict[str, dict[str, Any]]
    source_path: Path
    signature: str


def default_feature_build_config_path() -> Path:
    return DEFAULT_FEATURE_BUILD_CONFIG_PATH


def _validate_raw(doc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise ValueError("feature config root must be a JSON object")
    feats = doc.get("features")
    if not isinstance(feats, dict):
        raise ValueError("feature config must contain a 'features' object")
    for key in FEATURE_GROUP_KEYS:
        if key not in feats:
            raise ValueError(f"feature config missing group {key!r}")
        entry = feats[key]
        if not isinstance(entry, dict):
            raise ValueError(f"feature group {key!r} must be an object")
        if "enabled" not in entry:
            raise ValueError(f"feature group {key!r} must have 'enabled'")
        if not isinstance(entry["enabled"], bool):
            raise ValueError(f"feature group {key!r}: 'enabled' must be a boolean")
        for opt in ("description", "cost"):
            if opt in entry and not isinstance(entry[opt], str):
                raise ValueError(f"feature group {key!r}: {opt!r} must be a string when present")
    if not feats["base_transaction"]["enabled"]:
        raise ValueError("base_transaction cannot be disabled (required for the pipeline)")
    return feats


def load_feature_build_config(path: Path) -> LoadedFeatureBuildConfig:
    """Load JSON from ``path`` and return flags plus metadata for the build manifest."""
    raw_text = path.read_text(encoding="utf-8")
    doc = json.loads(raw_text)
    feats = _validate_raw(doc)

    flags = FeatureBuildFlags(
        base_transaction=bool(feats["base_transaction"]["enabled"]),
        corridor_risk_score=bool(feats["corridor_risk_score"]["enabled"]),
        weekly_internal_graph=bool(feats["weekly_internal_graph"]["enabled"]),
        rolling_account_activity=bool(feats["rolling_account_activity"]["enabled"]),
        company_age_days_proxy=bool(feats["company_age_days_proxy"]["enabled"]),
        pep_proxy_synthetic=bool(feats["pep_proxy_synthetic"]["enabled"]),
        hrj_country_flag=bool(feats["hrj_country_flag"]["enabled"]),
        account_daily=bool(feats["account_daily"]["enabled"]),
    )
    groups_meta: dict[str, dict[str, Any]] = {}
    for key in FEATURE_GROUP_KEYS:
        e = feats[key]
        groups_meta[key] = {
            "enabled": e["enabled"],
            "description": e.get("description", ""),
            "cost": e.get("cost", ""),
        }
    enabled_only = {k: groups_meta[k]["enabled"] for k in FEATURE_GROUP_KEYS}
    sig = hashlib.sha256(json.dumps(enabled_only, sort_keys=True).encode()).hexdigest()[:16]
    return LoadedFeatureBuildConfig(
        flags=flags,
        groups_meta=groups_meta,
        source_path=path.resolve(),
        signature=sig,
    )
