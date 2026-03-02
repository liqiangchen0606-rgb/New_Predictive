# MISTAKE LOG

This file records AI-generated mistakes that were caught and corrected during the coursework workflow. It is intended as appendix evidence showing that agent outputs were verified rather than accepted uncritically.

## ML-001: EDA labels used raw feature codes

- Stage: Section 2
- Mistake: Early EDA charts used raw column codes as titles and axis labels, which made the figures hard to interpret.
- How it was caught: Manual review identified that markers would not understand feature abbreviations without business-readable labels.
- Correction: Plot labels were replaced with human-readable variable names and clearer category labels.
- Why it matters: This improved interpretability and made the EDA defensible in the report.

## ML-002: Metric framing initially leaned too heavily toward ROC-AUC

- Stage: Section 4
- Mistake: The initial framing leaned too strongly toward ROC-AUC as the main evaluation lens.
- How it was caught: Manual review challenged whether that matched the churn-ranking objective.
- Correction: PR-AUC (Average Precision) was kept as the primary model-selection metric, with ROC-AUC and F1 retained as secondary checks.
- Why it matters: This aligned evaluation with the business objective of identifying likely churners while still reporting complementary discrimination metrics.

## ML-003: Lambda-based preprocessing broke model serialization

- Stage: Section 3 / Section 5
- Mistake: The preprocessing design used notebook-local lambda transformers.
- How it was caught: `joblib.dump` failed with a `PicklingError` when the tuned pipeline was saved.
- Correction: The lambda-based steps were replaced with sklearn-native preprocessing branches for numeric, object-categorical, and binary-flag features.
- Why it matters: The final pipeline became serializable, reproducible, and suitable for deployment-style artifact saving.

## ML-004: Section 3 default pipeline was inconsistent with ablation evidence

- Stage: Section 4
- Mistake: The default Section 3 pipeline still retained engineered features even after ablation showed they should not be part of the default final path.
- How it was caught: Manual review identified that the feature-set evidence and the actual downstream pipeline had diverged.
- Correction: The default final pipeline was reset to the dropped-feature, no-engineering baseline, while engineered features were kept only as a tested candidate branch.
- Why it matters: This ensured the final benchmark, tuning, and evaluation all used a feature set supported by the ablation evidence.

## ML-005: Initial ablation design was weaker than a like-for-like comparison

- Stage: Section 4
- Mistake: The early ablation setup was tied too narrowly to a single reference-model configuration, which made interpretation weaker than a broader model-family comparison.
- How it was caught: Manual review questioned whether the feature-set tests were fully aligned with the model-comparison logic.
- Correction: The ablation was reframed as a clearer feature-set decision step and expanded to reuse the same model families as the Section 4 comparison table.
- Why it matters: This made the feature-set conclusions less dependent on one estimator and more defensible as controlled experimental evidence.

## ML-006: Tracking chronology initially implied a return to Section 4 after testing

- Stage: Tracking / Appendix evidence
- Mistake: The tracking log order made it look as if the workflow went back to Section 4 after the later evaluation stages.
- How it was caught: Manual review identified that the chronology would confuse a marker reading the appendix evidence.
- Correction: The relevant entries were moved and renumbered so the Section 4 feature-set refinement and final benchmark rerun appear before Section 5.
- Why it matters: The appendix now reflects the true workflow: feature-set correction happened before final tuning and held-out evaluation.

## ML-007: README and script defaults drifted away from the actual dataset path

- Stage: Repository documentation / CLI defaults
- Mistake: The repo documentation and script defaults still pointed to an older CSV filename while the notebook used `data/raw/Telecom_customer churn.csv`.
- How it was caught: Manual repo review found that README instructions and `src` defaults no longer matched the actual submission data file.
- Correction: README and the relevant `src` defaults were updated to the real dataset path.
- Why it matters: This removed reproducibility friction for markers using the repo outside the notebook.
