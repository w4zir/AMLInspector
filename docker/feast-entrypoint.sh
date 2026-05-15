#!/usr/bin/env bash
set -euo pipefail
export FEAST_FS_YAML_FILE_PATH="${FEAST_FS_YAML_FILE_PATH:-/app/feast_repo/feature_store.docker.yaml}"
cd /app/feast_repo
python /app/docker/ensure_feast_postgres_databases.py
python /app/docker/bootstrap_feast_parquets.py
feast apply
exec feast serve --host 0.0.0.0 --port 6566
