"""Project paths and environment-driven service URLs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
FEAST_REPO = PROJECT_ROOT / "feast_repo"


def feast_online_store_url() -> str:
    """Postgres URL for Feast online store (matches docker compose credentials)."""
    if url := os.environ.get("FEAST_ONLINE_STORE_URL"):
        return url
    host = os.environ.get("FEAST_POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("FEAST_POSTGRES_PORT", "5432")
    user = os.environ.get("FEAST_POSTGRES_USER", "aml")
    password = os.environ.get("FEAST_POSTGRES_PASSWORD", "aml")
    database = os.environ.get("FEAST_POSTGRES_DATABASE", "feast")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def mlflow_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
