# Submission Report - Day21 Track2 CI/CD for AI Systems

## 1) Scope Completed

Completed items:

- Step 3 continuous training flow with new data (`add_new_data.py` -> DVC update -> retrain path)
- Bonus 2: multi-model training support (`random_forest`, `gradient_boosting`, `logistic_regression`)
- Bonus 3: automatic performance report (`outputs/report.txt`) and artifact upload
- Bonus 4: rollback guard in CI (compare new accuracy vs currently deployed accuracy in cloud)
- Bonus 5: label-distribution drift warning + metrics logging to `outputs/metrics.json`
- Bonus 1 readiness: optional DagsHub MLflow integration in workflow + setup guide

## 2) Step 3 Verification

### 2.1 Data growth

- Before: `train_phase1.csv` had `2998` rows.
- Ran `python add_new_data.py`.
- After: `train_phase1.csv` has `5996` rows.

### 2.2 DVC versioning and remote push

- Initialized DVC repo and remote (`s3://lab-21/dvc`, endpoint `sgp1.digitaloceanspaces.com`)
- Added DVC pointers:
  - `data/train_phase1.csv.dvc`
  - `data/eval.csv.dvc`
  - `data/train_phase2.csv.dvc`
- Pushed data to remote with `dvc push`.

### 2.3 Metric comparison table (required by Step 3.6)

Local measured values:

| Metric | Step 2 (2998 rows) | Step 3 (5996 rows) |
|---|---:|---:|
| accuracy | 0.6900 | 0.7480 |
| f1_score | 0.6893 | 0.7470 |

Evidence files:

- `submission/metrics_step2_local.json`
- `submission/metrics_step3_local.json`

## 3) Bonus Task Verification

### Bonus 1 - DagsHub MLflow tracking

Implemented workflow support and fallback logic:

- If `DAGSHUB_*` secrets exist -> train job logs to DagsHub.
- If not set -> local MLflow tracking fallback.

Guide created: `submission/dagshub_mlflow_setup.md`.

### Bonus 2 - Multiple algorithms

Implemented in `src/train.py`:

- `model_type: random_forest | gradient_boosting | logistic_regression`
- Safe model-specific parameter filtering

### Bonus 3 - Automatic performance report

Implemented in `src/train.py` + workflow:

- Confusion matrix text output
- Per-class precision and recall
- Saved to `outputs/report.txt`
- Uploaded together with `outputs/metrics.json` as artifact `train-artifacts`

### Bonus 4 - Rollback guard

Implemented in workflow:

- Eval job downloads previous deployed metrics from cloud key:
  - `metrics/latest/metrics.json`
- Compares new vs old `accuracy`
- Blocks deploy if new accuracy is lower

### Bonus 5 - Label distribution warning

Implemented in `src/train.py`:

- Computes class distribution for labels `0,1,2`
- Warns if any class ratio `< 0.10`
- Stores distribution and warnings in `outputs/metrics.json`

## 4) Repo Reorganization and Hygiene

Completed:

- Added stronger ignore rules in `.gitignore`:
  - `mlruns/`, nested `__pycache__/`, pytest temp/cache folders, temp key file
- Removed leftover temp key file from workspace
- Added `submission/` deliverables:
  - `report.md`
  - `dagshub_mlflow_setup.md`
  - metric/report evidence snapshots

## 5) Notes and Known Limits

- On local Python 3.13, full `pip install -r requirements.txt` can fail because `mlflow==2.13.0` pulls `pyarrow<16` (wheel/build compatibility issue). We used the already-installed global env and only updated `pathspec` for DVC compatibility.
- For final grading screenshots, capture:
  - GitHub Actions run showing all jobs green
  - `/health` and `/predict` responses
  - Cloud storage paths for DVC data + model + metrics
