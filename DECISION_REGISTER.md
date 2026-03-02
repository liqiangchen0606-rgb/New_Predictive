# DECISION REGISTER

This file tracks the key modelling and workflow decisions made during the coursework, in the order they were made.

## DR-001: Single source notebook

- Date: 2026-02-26
- Decision: Keep `MSIN0097_IndividualA.ipynb` as the only main notebook.
- Rationale: Avoid divergence between duplicate notebooks and keep the full narrative, outputs, and evidence in one place.
- Owner verification: Manual check that all required coursework sections exist in this notebook.

## DR-002: Data acquisition by manual local download

- Date: 2026-02-26
- Decision: Use a manually downloaded Kaggle CSV stored locally rather than calling the Kaggle API inside the notebook.
- Rationale: Keeps the notebook reproducible in an offline marking environment and avoids adding an external runtime dependency to the submission workflow.
- Owner verification: Manual check that the notebook loads the local file from `data/raw/Telecom_customer churn.csv`.

## DR-003: Leakage-safe preprocessing design

- Date: 2026-02-26
- Decision: Use a `sklearn` `Pipeline` plus `ColumnTransformer`.
- Rationale: Ensures transforms are fit on training data only, reducing leakage risk and making preprocessing reproducible.
- Owner verification: Manual review of preprocessing flow and split discipline in the notebook and reusable `src` pipeline.

## DR-004: Retain zero values as valid behavior

- Date: 2026-02-26
- Decision: Treat zeros as valid non-usage values rather than recoding them as missing.
- Rationale: In this telecom dataset, many zero-heavy fields represent genuine inactivity or no usage, which can still be predictive.
- Owner verification: Manual review of zero-inflation patterns in Section 2 and confirmation that zeros are preserved in Section 3.

## DR-005: Metric hierarchy for model selection

- Date: 2026-02-26
- Decision: Use PR-AUC as the primary metric, with ROC-AUC and F1 as secondary metrics.
- Rationale: PR-AUC aligns best with the business objective of identifying churners, while ROC-AUC and F1 provide complementary checks on discrimination and thresholded performance.
- Owner verification: Manual review of Section 4 comparison logic and final metric interpretation.

## DR-006: Treat calibration as diagnostic, not a separate optimization step

- Date: 2026-02-26
- Decision: Plot and discuss calibration, but do not apply formal recalibration in the final pipeline.
- Rationale: The calibration curve was useful as a probability-quality diagnostic, but adding calibration methods would increase workflow complexity beyond what was necessary for the final coursework submission.
- Owner verification: Manual check that calibration is reported in Section 5/Section 6 and that no recalibration step is inserted into the pipeline.

## DR-007: Final feature-set selection after ablation

- Date: 2026-02-26
- Decision: Keep the selected high-missing feature drop, but do not retain engineered features in the default final pipeline.
- Rationale: Section 4 ablation tests were used to validate feature-set choices before the final benchmark was locked; engineered features remained a documented candidate branch, but the final pipeline was aligned to the dropped-feature baseline after the feature-set review.
- Owner verification: Manual check that the final artifacts were regenerated with `engineered_features_used_in_final_pipeline` recorded as `false` in `artifacts/feature_schema.json`.


## DR-008: Shortlist the top two model families for tuning

- Date: 2026-02-26
- Decision: Carry `XGBoost` and `HistGradientBoosting` into Section 5 tuning.
- Rationale: They were the strongest model families in Section 4 under cross-validated PR-AUC and therefore justified deeper tuning.
- Owner verification: Manual review of `artifacts/model_comparison.csv` and the Section 4 shortlist output.

## DR-009: Keep a neutral threshold of 0.50

- Date: 2026-02-26
- Decision: Use a default classification threshold of `0.50` in the final reported workflow.
- Rationale: Threshold trade-offs were inspected using out-of-fold training probabilities, but the final cut-off was kept at the standard neutral default so the operational threshold can later be chosen by the firm based on business cost and capacity.
- Owner verification: Manual check that `artifacts/threshold.json` and final reported metrics both use `0.50`.