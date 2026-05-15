"""Tests for :mod:`aml_inspector.features.feature_build_config`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aml_inspector.features.feature_build_config import load_feature_build_config


def test_load_feature_build_config_rejects_disabled_base(tmp_path: Path) -> None:
    p = tmp_path / "fc.json"
    p.write_text(
        json.dumps(
            {
                "features": {
                    "base_transaction": {"enabled": False, "description": "x", "cost": "cheap"},
                    "corridor_risk_score": {"enabled": True, "description": "x", "cost": "cheap"},
                    "weekly_internal_graph": {"enabled": True, "description": "x", "cost": "cheap"},
                    "rolling_account_activity": {"enabled": True, "description": "x", "cost": "cheap"},
                    "company_age_days_proxy": {"enabled": True, "description": "x", "cost": "cheap"},
                    "pep_proxy_synthetic": {"enabled": True, "description": "x", "cost": "cheap"},
                    "hrj_country_flag": {"enabled": True, "description": "x", "cost": "cheap"},
                    "account_daily": {"enabled": True, "description": "x", "cost": "cheap"},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_transaction cannot be disabled"):
        load_feature_build_config(p)


def test_load_feature_build_config_requires_all_groups(tmp_path: Path) -> None:
    p = tmp_path / "fc.json"
    p.write_text(json.dumps({"features": {"base_transaction": {"enabled": True}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing group"):
        load_feature_build_config(p)
