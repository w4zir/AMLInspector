# AMLInspector

Models and workflows to detect money laundering (AML), using **Feast** for features and **MLflow** for experiment tracking. Data layout targets the Kaggle [IBM Transactions for Anti-Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) dataset.

## Layout

- `data/raw/` — Kaggle CSVs (contents gitignored; keep `.gitkeep`).
- `data/interim/` — cleaned or sampled tables for iteration.
- `data/processed/` — Parquet/CSVs for modeling and Feast `FileSource` paths (includes a small sample Parquet for the scaffold).
- `feast_repo/` — Feast project (`feature_store.yaml`, entities, sources, feature views). Run Feast CLI commands from this directory.
- `src/aml_inspector/` — importable package: config, data helpers, Feast glue, training/eval scripts.
- `notebooks/` — EDA and ad-hoc Feast retrieval.
- `docker/` — Compose stack for Postgres, MLflow, and the Feast HTTP feature server.

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional; defaults match compose
```

## Docker (Postgres + MLflow + Feast feature server)

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

- **Postgres**: `localhost:5432`, user `aml`, password `aml`, databases `feast` and `mlflow` (created by `docker/init-databases.sh`).
- **MLflow UI**: http://127.0.0.1:5000 — backend `mlflow` DB, artifacts under the `mlflow_artifacts` volume.
- **Feast feature server**: http://127.0.0.1:6566 — on startup runs `feast apply` then `feast serve` using `feast_repo/feature_store.docker.yaml` (Postgres host `postgres`).

To apply the feature store from your host (uses `feast_repo/feature_store.yaml` with `127.0.0.1`):

```bash
docker compose -f docker/docker-compose.yml up -d postgres
cd feast_repo && feast apply
```

## Kaggle data

Download into `data/raw/`, for example:

```bash
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/raw --unzip
```

Or use the bundled helper (requires [Kaggle API credentials](https://www.kaggle.com/docs/api) and `pip install kaggle` so the `kaggle` CLI is on your `PATH`):

```bash
python -m aml_inspector.data.kaggle_download
```

This fetches `HI-Small_Trans.csv` and `HI-Medium_Trans.csv` by default. Use `python -m aml_inspector.data.kaggle_download --help` for options.

### Home-bank subset (sender or receiver)

After the CSVs are in `data/raw/`, filter to a single bank that appears as **From Bank** or **To Bank**, auto-picking a bank with strong counts of both laundering and non-laundering involvement (override with `--bank-id`):

```bash
python -m aml_inspector.data.preprocess_home_bank \
  --input-files HI-Small_Trans.csv HI-Medium_Trans.csv
```

Outputs:

- `data/processed/home_bank_transactions.parquet` — filtered rows plus `event_timestamp` (from `Timestamp`) when present
- `data/interim/home_bank_selection_summary.json` — chosen bank id and counts

Then align entity and timestamp column names in `src/aml_inspector/data/datasets.py` and `feast_repo/entities.py` after EDA in `notebooks/01_eda_ibm_aml.ipynb`.

## Training (MLflow smoke)

With the MLflow server running:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m aml_inspector.modeling.train
```

## Tests

```bash
pytest
```
