# WORKFLOW EVIDENCE

All notable coursework-repo changes are recorded here.

## 2026-02-26

### Added

- Core pipeline modules:
  - `src/data_io.py`
  - `src/preprocess.py`
  - `src/train.py`
  - `src/evaluate.py`
  - `src/utils.py`
- Smoke test scaffold: `tests/tests_smoke.py`
- Project setup docs: `README.md`, `requirements.txt`
- Repository hygiene: `.gitignore`

### Updated

- `requirements.txt`:
  - added `seaborn>=0.13`
  - added `xgboost>=2.0` (optional Section 4 model)
- `README.md`:
  - added macOS note for XGBoost/OpenMP dependency (`brew install libomp`)
  - documented fallback behavior when XGBoost cannot load
- `MSIN0097_IndividualA.ipynb`:
  - Added coursework-aligned section structure.
  - Implemented Section 1 content:
    - target/type definition
    - metric strategy (PR-AUC primary; ROC-AUC and F1 secondary)
    - constraints
    - assumptions/limitations
    - Codex delegation vs manual verification plan
  - Added configurable local dataset-loading code cell (no Kaggle API calls).
  - Updated loader to auto-detect CSV files under `data/` and `data/raw/`.
  - Added explicit fallback for `data/raw/Telecom_customer churn.csv`.
  - Implemented Section 2 EDA workflow:
    - minimal non-leaky `df_eda` cleaning
    - missingness summary (count + %), churn rate, leakage-term detection
    - figure exports to `artifacts/figures/`
    - data quality issues subsection
    - corrected chart communication after user feedback:
      - replaced raw variable codes in plot titles/x-axis with human-readable variable meanings
  - Implemented Section 3 preparation workflow:
    - target definition and binary enforcement (`churn`)
    - stratified 80/20 train/test split (`random_state=42`)
    - programmatic numeric/categorical typing with binary flag reassignment
    - dropped ID-like column (`customer_id`) before preprocessing
    - dropped selected high-missing lower-priority fields during preparation (`numbcars`, `dwllsize`, `hhstatin`, `ownrent`, `dwlltype`, `infobase`)
    - leakage-safe preprocessing pipeline fit on train only
    - saved `artifacts/feature_schema.json` and `artifacts/split_config.json`
    - added explanatory note for transformed feature dimension growth after preprocessing
    - corrected serialization bug by removing lambda-based transformers and replacing them with sklearn-native preprocessing branches for numeric, object-categorical, and binary-flag features
  - Implemented Section 4 model exploration workflow:
    - model zoo with baseline + multiple model families (plus optional XGBoost)
    - 5-fold stratified cross-validation on training split only
    - metrics: PR-AUC (primary), ROC-AUC, F1 + runtime logging
    - saved ranked comparison to `artifacts/model_comparison.csv`
    - identified top-2 shortlist candidates for Section 5
    - corrected metric framing after review:
      - PR-AUC kept as primary for business alignment (identifying churners)
      - ROC-AUC retained as complementary reporting metric because the dataset is relatively balanced
      - noted that lower real-world churn prevalence could reduce absolute PR-AUC / precision
  - Implemented Section 5 tuning/evaluation workflow:
    - training-only hyperparameter search, threshold tuning, and one-time final test evaluation
    - saved tuning outputs, model artifact, threshold artifact, metrics, plots, and error analysis
  - Revised Section 5 for computational constraints:
    - tuned XGBoost only
    - initially reduced `RandomizedSearchCV` to 12 iterations, then increased to a moderate 18-iteration search
    - used 3-fold CV for tuning and out-of-fold threshold predictions
    - documented this as a runtime-conscious compromise
  - Implemented Section 6 final solution workflow:
    - replaced the placeholder checklist with an artifact-driven narrative summary (no rerun of modelling)
    - loaded saved metrics, threshold, tuning, and split artifacts to populate final model summary and business interpretation
    - added deployment plan, model risk/limitation review, fairness considerations, improvement roadmap, and concluding paragraph
    - refined notebook presentation by hiding the summary index and showing full wrapped text in Section 6 narrative tables
- Data verification:
  - Confirmed successful read from `data/raw/Telecom_customer churn.csv`.
  - Observed loaded schema shape `(100000, 100)` for subsequent target-column confirmation.

### Removed

- `notebooks/churn_pipeline.ipynb` (to keep a single main notebook).
