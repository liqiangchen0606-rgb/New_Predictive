# DECISION REGISTER

This file tracks key implementation decisions and verification ownership for the coursework.

## DR-001: Single source notebook

- Date: 2026-02-26
- Decision: Keep `MSIN0097_IndividualA.ipynb` as the only main notebook.
- Rationale: Avoid divergence between duplicate notebooks and keep narrative/evidence in one place.
- Owner verification: Manual check that all six coursework headings exist in this notebook.

## DR-002: Leakage-safe preprocessing design

- Date: 2026-02-26
- Decision: Use `sklearn` `Pipeline` + `ColumnTransformer`.
- Rationale: Fit transforms on training data only; reduce leakage risk and improve reproducibility.
- Owner verification: Manual review of preprocessing flow and split discipline in `src/train.py`.

## DR-003: Metric hierarchy under class imbalance

- Date: 2026-02-26
- Decision: Use PR-AUC as primary metric; ROC-AUC and F1 as secondary.
- Rationale: PR-AUC is more informative when positive class is relatively rare.
- Owner verification: Manual interpretation of metrics and thresholding trade-offs in notebook.

## DR-004: Data acquisition handling

- Date: 2026-02-26
- Decision: Assume manual dataset download and local path configuration.
- Rationale: Keep notebook reproducible without requiring API calls during authoring.
- Owner verification: Manual check of file paths and schema after real CSV is placed.
