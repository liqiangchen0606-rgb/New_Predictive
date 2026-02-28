# TASKS

## Current Status

- [x] Repository scaffold created (`src/`, `tests/`, `artifacts/`, `data/`).
- [x] Main notebook set to `MSIN0097_IndividualA.ipynb`.
- [x] Section 1 (Problem framing) completed in notebook.
- [x] AI usage tracking file created (`AI_USAGE_LOG.md`).
- [x] Reproducible training/evaluation skeleton implemented.
- [x] Smoke test skeleton added (`tests/tests_smoke.py`).
- [x] Place real Kaggle CSV locally.
- [x] Update notebook loader to import from `data/` reliably.
- [x] Confirm target column and label encoding for downloaded dataset schema.
- [x] Complete Section 2 (EDA) with leakage and imbalance checks.
- [x] Revise Section 2 plot labels to human-readable variable meanings (user-corrected agent mistake).
- [x] Implement Section 3 preparation workflow (leakage-safe split + preprocessing + schema artifacts, including dropping selected high-missing low-priority fields).
- [x] Implement Section 4 CV model comparison workflow and shortlist logic.
- [x] Document corrected Section 4 metric rationale (PR-AUC primary, ROC-AUC complementary for balanced data).
- [x] Update dependency/docs for optional XGBoost execution (requirements + macOS note).
- [x] Implement Section 5 tuning/evaluation workflow with runtime-conscious simplification.
- [x] Document explicit agent-made bug and correction (serialization failure from lambda transformers).
- [x] Increase Section 5 tuning budget moderately (18 iterations, wider XGBoost search) while keeping runtime practical.
- [x] Run real training and save real artifacts in `artifacts/`.
- [x] Complete Sections 4-6 with evidence-based results and final narrative.
- [x] Build appendix-ready agent usage evidence and decision table.

## Immediate Next Step

- Draft the final 2000-word report PDF using the notebook outputs, saved artifacts, and tracking evidence.
