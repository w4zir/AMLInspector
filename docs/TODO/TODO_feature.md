# TODO: Feature generation (Feast feature store)

**MVP implemented:** run `python -m aml_inspector.pipelines.build_feature_data` (see [`README.md`](../../README.md)) to build split Medium/Small Parquets, feature tables, experiment entity frames, `data/interim/feature_build_manifest.json`, and optionally `feast apply`. Corridor + weekly internal graph artifacts are **fit on HI-Medium only**; velocity/fan-out use strictly prior events per `data_split_source`. **Deferred:** full point-in-time local clustering per event; nullable PII columns; wiring `train.py` to `get_historical_features` (see [`TODO_experimentation.md`](TODO_experimentation.md)).

This checklist implements the **Feature Engineering Layer** and **Data Preprocessing** rules from [`docs/specs/data_model_spec.md`](../specs/data_model_spec.md), using **Feast** as defined in [`README.md`](../../README.md) and the scaffold under [`feast_repo/`](../../feast_repo/).

**Data split policy (modeling):** Use **HI-Medium** only for train/validation (80/20 after preprocessing). Use **HI-Small** only as a **holdout test** set. Feature pipelines must support this by **not leaking** Small statistics into Medium-derived features (see leakage section below).

---

## 0. Prerequisites

- [ ] Raw CSVs present under `data/raw/` (e.g. `HI-Medium_Trans.csv`, `HI-Small_Trans.csv`) per Kaggle IBM AML layout.
- [ ] Python env: `pip install -e ".[dev]"` from repo root.
- [ ] Optional: Docker stack for Postgres + Feast + MLflow (`docker/docker-compose.yml`) when using online materialization or shared stores.
- [ ] Read [`docs/specs/data_model_spec.md`](../specs/data_model_spec.md) sections 2–3.

---

## 1. Validate inputs and column contract

- [ ] Confirm expected columns exist (align names with IBM CSV: e.g. `From Bank`, `To Bank`, `Is Laundering`, `Timestamp` — see [`src/aml_inspector/data/preprocess_home_bank.py`](../../src/aml_inspector/data/preprocess_home_bank.py)).
- [ ] Document final column rename map (if any) in [`src/aml_inspector/data/datasets.py`](../../src/aml_inspector/data/datasets.py) and keep Feast entities/sources in sync.
- [ ] Add **source tag** column (e.g. `data_split_source = medium | small`) on each row **before** any join, so train/val/test boundaries are auditable in Parquet.

---

## 2. Home Bank filter and visibility masking (spec §2.1)

- [ ] Choose **Home Bank** id (auto via [`python -m aml_inspector.data.preprocess_home_bank`](../../README.md) or fixed `--bank-id`).
- [ ] Filter rows where Home Bank appears as sender **or** receiver (existing helper writes `data/processed/home_bank_transactions.parquet` when both files are passed — **for modeling**, prefer **separate** Medium-only and Small-only Parquet outputs, or filter by `data_split_source` after a single run that tags source).
- [ ] **Visibility masking (legal / silo simulation):**
  - [ ] For transactions where the internal party is at Home Bank: retain usable internal identifiers for feature keys (subject to EDA: account-level entity id).
  - [ ] For **external** counterparties: do **not** use raw external account ids as learnable features; keep **Bank ID** and **Currency**; **hash** external account identifiers with a **stable salt per project** (store hash spec in `data/interim/` metadata only, not in Feast training labels).
- [ ] Log counts: rows, positives (`Is Laundering`), internal vs external-leg feature eligibility.

---

## 3. Cleaning and normalization (spec §2.2)

- [ ] **Temporal alignment:** parse `Timestamp` → UTC `event_timestamp` (Feast `timestamp_field` on batch sources).
- [ ] **Currency standardization:** convert amounts to a **base currency (USD)** using a documented rate table (fixed MVP vs historical — document choice in code/README).
- [ ] **Consistency:** ensure one row = one modeling event (transaction-level vs account-daily aggregates — pick one primary **entity** for Feast; see §6).

---

## 4. Feature tables (offline-first) under `data/processed/`

Produce **point-in-time-safe** Parquet tables keyed by entity + `event_timestamp`. Suggested minimal set (iterate names to match code):

| Table | Purpose | Primary entity key |
|-------|---------|-------------------|
| `txn_level_features.parquet` | Per-transaction features for corridor/velocity/roundness | e.g. `transaction_id` or composite key |
| `account_txn_features.parquet` | Account-scoped rolling / dwell features | `account_id` (internal) |
| `account_graph_features.parquet` | NetworkX-derived aggregates on **internal-only** subgraph | `account_id` |
| `account_daily_features.parquet` | Daily rollups (extends current Feast placeholder) | `account_id` |

- [ ] Ensure every feature row includes: `event_timestamp`, entity id(s), and **no future** columns (rolling windows must use **strictly prior** events only).
- [ ] For **HI-Small** holdout: any statistic that requires global distribution (e.g. corridor risk normalization) must be fit **only on HI-Medium train** and applied as a **frozen transform** to val/test; never refit on Small.

---

## 5. Feature domains (spec §3) — implementation steps

### A. Transactional and corridor

- [ ] **Velocity \(V_{24h}\):** count outbound txs per internal account in \((t-24h, t]\) (define outbound using Home Bank as origin).
- [ ] **Corridor risk score:** build `Source_Country` → `Target_Country` weight table (versioned JSON/CSV under `data/interim/`); join to each txn.
- [ ] **Dwell time:** for each internal account, \(\Delta t = t_{out} - t_{in}\) between paired inbound/outbound events (define pairing rules explicitly in code comments).
- [ ] **Amount roundness:** binary flag `amount % 100 == 0` on **USD-normalized** amount.

### B. Entity and community

- [ ] **Internal fan-out:** unique internal recipients / volume over a defined window (e.g. 7d, 30d — log window in Feast feature view description).
- [ ] **Shared PII linkage:** if IBM subset lacks device/IP, add **nullable** columns and populate from future internal tables; document “mock / N/A” for IBM-only MVP.
- [ ] **Local clustering coefficient:** build **internal-only** graph (nodes = internal accounts, edges = tx links); compute coefficient per node via NetworkX; snapshot at `event_timestamp` using **only edges with timestamp ≤ t** (expensive — start with weekly snapshot MVP).

### C. Public and external

- [ ] **HRJ status:** join counterparty bank country to FATF grey/black list (static reference table).
- [ ] **Company age proxy:** days since first observed txn for that account (or entity type flag if retail vs business unknown).
- [ ] **PEP match:** mock boolean from rules or random seed table **documented as synthetic** (do not present as real PEP screening).

---

## 6. Feast mapping (repo scaffold)

Paths: [`feast_repo/entities.py`](../../feast_repo/entities.py), [`feast_repo/data_sources.py`](../../feast_repo/data_sources.py), [`feast_repo/feature_views.py`](../../feast_repo/feature_views.py), [`feast_repo/feature_store.yaml`](../../feast_repo/feature_store.yaml).

- [ ] **Entities:** replace placeholder `account` join key with EDA-chosen id (e.g. internal account id after masking). Add `transaction` entity if modeling at txn level.
- [ ] **FileSource:** point each `FileSource` to the Parquet paths under `data/processed/`; set `timestamp_field` to `event_timestamp`.
- [ ] **FeatureView / Field:** declare schema dtypes (`Float32`, `Int64`, `Bool`, etc.) matching Parquet.
- [ ] **TTL:** set sensible TTL (e.g. 365d for aggregates) per view.
- [ ] **Online / offline:** enable `offline=True` for training retrieval; `online=True` only when serving path is defined.
- [ ] From repo root / `feast_repo`: run `feast apply` (see README). Fix import paths in Feast repo if needed (`data_sources` vs package layout).
- [ ] Optional: wire [`src/aml_inspector/features/materialize.py`](../../src/aml_inspector/features/materialize.py) or CLI docs for `feast materialize-incremental` once time ranges are fixed.

---

## 7. Training data retrieval (Feast)

- [ ] Build a **training point-in-time** dataset: entity id, `event_timestamp`, label `Is Laundering`, and all features from `get_historical_features` (or equivalent) for **HI-Medium train/val windows only**.
- [ ] For **HI-Small** test: run the **same** feature retrieval with timestamps from Small only, using **Medium-train-fitted** encoders/risk tables only.

---

## 8. Quality gates (leakage and schema)

- [ ] **Schema:** assert column set and dtypes match Feast `Field` definitions (CI test or notebook assertion).
- [ ] **Leakage:** verify no column uses future timestamps; verify global stats fit on train only.
- [ ] **Class balance:** log positive rate per split (`medium_train`, `medium_val`, `small_test`).
- [ ] **Nulls:** document imputation vs “unknown bucket” for graph/PII features.
- [ ] **Reproducibility:** log Parquet paths, git SHA, and Feast apply output hash in `data/interim/feature_build_manifest.json`.

---

## 9. Documentation handoff

- [ ] Update [`README.md`](../../README.md) with exact commands: preprocess → build features → `feast apply` → where training reads features.
- [ ] Cross-link [`docs/TODO/TODO_experimentation.md`](TODO_experimentation.md) for Champion-Challenger training on these features.

---

## Open points (resolve during implementation)

- **Entity grain:** transaction-level Champion vs account-level aggregates — spec spans both; pick primary Feast entity and document.
- **Graph cost:** full PIT graph features may be slow; scope MVP (e.g. degree only) vs full clustering coefficient.
- **External hashing:** ensure hashed ids are not reversible and are consistent across Medium and Small runs.
