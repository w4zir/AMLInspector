# TODO: Training and experimentation (Champion–Challenger + MLflow)

This checklist implements **§4 Modeling Architecture** and **§5 Implementation Roadmap** from [`docs/specs/data_model_spec.md`](../specs/data_model_spec.md), using **MLflow** as in [`README.md`](../../README.md) and extending beyond the current smoke trainer in [`src/aml_inspector/modeling/train.py`](../../src/aml_inspector/modeling/train.py).

**Dataset policy:**

| Split | Source file | Role |
|-------|-------------|------|
| Train | **HI-Medium** (Home Bank subset) | 80% — model fitting, threshold tuning, frozen transforms |
| Validation | **HI-Medium** (Home Bank subset) | 20% — hyperparameters, early stopping, Champion/Challenger **comparison** |
| Test (holdout) | **HI-Small** (Home Bank subset) | **Final** evaluation only — no tuning decisions |

Use **stratified** 80/20 on `Is Laundering` (or equivalent label column) **after** preprocessing and **within HI-Medium only**. **Never** tune on HI-Small.

---

## 0. Prerequisites

- [ ] Features available per [`TODO_feature.md`](TODO_feature.md) (Parquet + Feast `apply`).
- [ ] `MLFLOW_TRACKING_URI` set (default `http://127.0.0.1:5000` via [`src/aml_inspector/config.py`](../../src/aml_inspector/config.py)); MLflow UI up if using tracking server.
- [ ] Dependencies: `xgboost`, `scikit-learn` (Isolation Forest), optional VAE stack (e.g. PyTorch) if pursuing Challenger VAE path.

---

## 1. Experiment layout in MLflow

Create **separate experiments** (or nested runs with tags) for clarity:

| MLflow experiment (suggested name) | Purpose |
|-----------------------------------|---------|
| `aml_champion` | Supervised XGBoost runs (HI-Medium train/val) |
| `aml_challenger` | Unsupervised anomaly runs (same feature matrix, labels used **only** for evaluation) |
| `aml_champion_challenger` | Joint policy evaluation, calibration, and holdout reporting |

**Run tags (every run):**

- [ ] `split_policy`: `medium_80_20_small_holdout`
- [ ] `data_revision`: git SHA + feature manifest id
- [ ] `home_bank_id`: chosen bank
- [ ] `model_role`: `champion` | `challenger` | `ensemble_policy`

---

## 2. Data loading and splits

- [ ] Load **HI-Medium** Home Bank Parquet (see [`src/aml_inspector/data/preprocess_home_bank.py`](../../src/aml_inspector/data/preprocess_home_bank.py)); drop or ignore any row tagged as `small` if present in combined files.
- [ ] **Stratified 80/20** train/validation split with **fixed** `random_state`; log row counts and positive rate per split to MLflow.
- [ ] Load **HI-Small** Home Bank Parquet as **test**; log counts once; **do not** compute thresholds from Small.
- [ ] Log **Feast dataset** or artifact references (e.g. path to training frame Parquet) via `mlflow.log_input` / artifacts when using MLflow 2.x dataset tracking.

---

## 3. Champion (supervised XGBoost) — spec §4.1

### 3.1 Training loop

- [ ] Build feature matrix `X` and label `y` from Feast offline retrieval for **train** only; same feature columns for val/test.
- [ ] Handle imbalance: `scale_pos_weight` ≈ `neg_count / pos_count` on **train**; log exact value.
- [ ] Use **validation** for: early stopping rounds, `eval_metric` (e.g. `aucpr`, `logloss`).
- [ ] Log params: `max_depth`, `eta`, `subsample`, `colsample_bytree`, `min_child_weight`, `scale_pos_weight`, `seed`, feature list version.

### 3.2 Threshold and AML-oriented metrics (spec §5 Phase 5)

- [ ] Default scoring: predicted probability of positive class.
- [ ] On **validation**: sweep thresholds; compute confusion matrix, precision/recall, **cost-weighted** metric (define `C_miss`, `C_false_alarm` in code constants and log them).
- [ ] Choose **operating threshold** using validation only; freeze for Small test.
- [ ] Log curves: PR curve, optional ROC as artifact.

### 3.3 Artifacts and model registry

- [ ] `mlflow.xgboost.log_model` (or `pyfunc` wrapper) with **signature** (input schema).
- [ ] Register candidate as `ChampionCandidate` in MLflow Model Registry with stage `Staging`; promote to `Production` only after Small test gate (§7).

---

## 4. Challenger (unsupervised) — spec §4.2

### 4.1 Isolation Forest (recommended first)

- [ ] Train on **train split features only** (optionally **exclude** label-derived columns if any); **do not** fit on validation rows to avoid optimistic bias if using unsupervised scores for model selection — *for IF, fitting on train-only is acceptable; evaluate on val using `decision_function` or normalized score*.
- [ ] Log params: `n_estimators`, `max_samples`, `contamination` (if used), `random_state`.
- [ ] Define **anomaly score** → percentile or calibrated flag; log score distributions on val (artifact: histogram).

### 4.2 Optional VAE path

- [ ] Separate experiment branch; log architecture, latent dim, epochs, learning rate.
- [ ] Define reconstruction error as anomaly score; document training stability and compute cost.

### 4.3 Challenger evaluation (uses labels **only** for metrics, not training)

- [ ] On validation: rank by anomaly score; report precision@k, recall of positives in top-k%, overlap with Champion misses.
- [ ] **Primary product metric (spec):** count / rate of **Challenger-positive & Champion-negative** (high-priority review queue); log as `challenger_only_flag_rate` and `challenger_only_precision` if labels exist.

---

## 5. Champion–Challenger combined policy

Define explicit rules (tunable on **validation** only):

- [ ] **Alert if:** `Champion_score >= T_champ` **OR** `Challenger_score >= T_chall` **OR** `(Challenger_score >= T_chall AND Champion_score < T_champ_soft)` for “Champion miss” queue.
- [ ] Log `T_champ`, `T_chall`, and policy version string.
- [ ] Log **confusion-style** outcomes for the **combined** policy on validation.
- [ ] Document manual review simulation: cap review budget `B` (max alerts per day); log recall under budget.

---

## 6. Holdout evaluation on HI-Small (test)

- [ ] Apply **frozen** Champion model + **frozen** thresholds from Medium validation.
- [ ] Apply **frozen** Challenger (fitted on Medium **train** only, never on Small).
- [ ] Log **all** primary metrics on Small once; no retuning.
- [ ] If Small is too low-N for stable PR-AUC, log confidence intervals (bootstrap) as artifact.

---

## 7. Promotion criteria (Champion / Challenger acceptance)

Document in MLflow run description:

- [ ] **Champion:** validation PR-AUC / recall at fixed precision ≥ agreed bar; Small test recall at operational precision ≥ bar-ε.
- [ ] **Challenger:** non-trivial lift on `challenger_only` positives on validation; no explosion of false alerts on Small under budget `B`.
- [ ] **Registry:** only one `Production` Champion version at a time; Challenger can be `Staging` until shadow period defined.

---

## 8. Extend project entrypoints

- [ ] Replace [`src/aml_inspector/modeling/train.py`](../../src/aml_inspector/modeling/train.py) smoke test with CLI: `--role champion|challenger`, `--split medium`, paths, experiment name.
- [ ] Extend [`src/aml_inspector/modeling/evaluate.py`](../../src/aml_inspector/modeling/evaluate.py) to load a registered run, score val/test, log metrics and plots.
- [ ] Add `pytest` for split integrity (no Small rows in train), label stratification, and MLflow param presence (optional integration test with local file store).

---

## 9. Reproducibility checklist

- [ ] Single **config YAML** or dataclass logged as MLflow artifact (thresholds, paths, seeds).
- [ ] Record `feast feature-view list` output snapshot after `feast apply`.
- [ ] Document human steps vs automated steps in [`README.md`](../../README.md).

---

## Cross-references

- Feature pipeline and leakage rules: [`TODO_feature.md`](TODO_feature.md)
- Product spec: [`docs/specs/data_model_spec.md`](../specs/data_model_spec.md)
