"""Create Feast / MLflow databases on Postgres if missing (idempotent).

Older Docker volumes may have been initialized before ``init-databases.sh`` ran or
when that script had CRLF line endings and failed under Linux.
"""

from __future__ import annotations

import os
import time

import psycopg

_ADMIN_URL = os.environ.get(
    "FEAST_POSTGRES_ADMIN_URL",
    "postgresql://aml:aml@postgres:5432/postgres",
)
_DATABASES = ("feast", "mlflow")


def main() -> None:
    deadline = time.monotonic() + 120
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = psycopg.connect(_ADMIN_URL, connect_timeout=5)
            break
        except Exception as e:
            last = e
            time.sleep(2)
    else:
        raise RuntimeError(f"Postgres not reachable at {_ADMIN_URL!r}") from last

    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for name in _DATABASES:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (name,),
                )
                if cur.fetchone() is None:
                    cur.execute(f"CREATE DATABASE {name}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
