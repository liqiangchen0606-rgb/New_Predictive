# MSIN0097 Churn Prediction Pipeline

Reproducible coursework scaffold for telecom churn prediction using Kaggle dataset `abhinav89/telecom-customer`.

## Project Structure

- `MSIN0097_IndividualA.ipynb`: main notebook with required 6 coursework headings.
- `src/data_io.py`: data loading and target split helpers.
- `src/preprocess.py`: leakage-safe `ColumnTransformer` + preprocessing.
- `src/train.py`: training, model comparison, artifact persistence.
- `src/evaluate.py`: evaluation utilities and saved-model evaluation CLI.
- `src/utils.py`: shared utilities (seed, logging, JSON save, directories).
- `tests/tests_smoke.py`: smoke test for one-model train run and artifact creation.
- `artifacts/`: output directory for run artifacts.

## Install

1. Create and activate virtual environment.
2. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### XGBoost Note (macOS)

Section 4 includes XGBoost benchmarking. On macOS, XGBoost may require OpenMP runtime:

```bash
brew install libomp
```

If XGBoost cannot load, the notebook fallback continues with non-XGBoost models.

## Data Download (Kaggle)

Expected file location:

- `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Example using Kaggle CLI:

```bash
kaggle datasets download -d abhinav89/telecom-customer -p data/raw --unzip
```

If the extracted file has a different name, rename it to:

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Run Training

From repo root:

```bash
python -m src.train --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --artifact-dir artifacts
```

Optional model subset:

```bash
python -m src.train --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --models logistic_regression
```

Optional missingness ablation (drop columns above threshold):

```bash
python -m src.train --data-path data/raw/Telecom_customer\\ churn.csv --models logistic_regression --drop-high-missing --missing-threshold 0.40
```

## Run Evaluation

```bash
python -m src.evaluate --model-path artifacts/best_model.joblib --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --artifact-dir artifacts
```

## Smoke Test

```bash
pytest -q tests/tests_smoke.py
```

## Artifacts

Training/evaluation writes to:

- `artifacts/metrics.json`
- `artifacts/model_comparison.csv`
- `artifacts/best_model.joblib`
- `artifacts/figures/` (e.g., confusion matrix PNG)

## Tracking Files

- `TASKS.md`: execution checklist and next actions.
- `WORKFLOW_EVIDENCE.md`: dated record of code/notebook changes and workflow evidence.
- `DECISION_REGISTER.md`: key modelling/process decisions and verification ownership.
- `AI_USAGE_LOG.md`: evidence log of AI-assisted work and manual verification decisions.

## Notes

- This repository is a runnable skeleton with TODO markers for dataset-specific decisions and deeper analysis.
- No fabricated performance results are included.
