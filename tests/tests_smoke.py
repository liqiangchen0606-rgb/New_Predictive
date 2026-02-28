"""Smoke test for churn pipeline reproducibility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_io import ensure_project_layout
from src.train import train_and_select


def _build_sample_dataset(n_rows: int = 120) -> pd.DataFrame:
    """Create a tiny synthetic Telco-like dataset for smoke testing."""
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "customerID": f"CUST-{i:05d}",
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": int(i % 7 == 0),
                "Partner": "Yes" if i % 3 == 0 else "No",
                "Dependents": "No" if i % 4 else "Yes",
                "tenure": (i % 60) + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No" if i % 5 else "Yes",
                "InternetService": "Fiber optic" if i % 3 == 0 else "DSL",
                "OnlineSecurity": "No" if i % 4 else "Yes",
                "OnlineBackup": "Yes" if i % 2 == 0 else "No",
                "DeviceProtection": "Yes" if i % 3 else "No",
                "TechSupport": "No" if i % 6 else "Yes",
                "StreamingTV": "Yes" if i % 2 else "No",
                "StreamingMovies": "No" if i % 3 else "Yes",
                "Contract": "Month-to-month" if i % 2 else "Two year",
                "PaperlessBilling": "Yes" if i % 2 else "No",
                "PaymentMethod": "Electronic check" if i % 3 == 0 else "Mailed check",
                "MonthlyCharges": 20.0 + (i % 50) * 1.1,
                "TotalCharges": 20.0 + (i % 50) * 1.1 * ((i % 60) + 1),
                "Churn": "Yes" if i % 4 == 0 else "No",
            }
        )
    return pd.DataFrame(rows)


def test_smoke_train_one_model_and_write_artifacts(tmp_path: Path) -> None:
    base = tmp_path
    ensure_project_layout(base)

    data_path = base / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    artifacts_path = base / "artifacts"
    df = _build_sample_dataset()
    df.to_csv(data_path, index=False)

    result = train_and_select(
        data_path=data_path,
        artifact_dir=artifacts_path,
        model_names=["logistic_regression"],
    )

    assert result.best_model_name == "logistic_regression"
    assert (artifacts_path / "metrics.json").exists()
    assert (artifacts_path / "model_comparison.csv").exists()
    assert (artifacts_path / "best_model.joblib").exists()
    assert (artifacts_path / "figures").exists()

