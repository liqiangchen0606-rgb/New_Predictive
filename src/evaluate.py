"""Evaluation utilities and artifact generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import data_io
from src.preprocess import prepare_feature_frame
from src.utils import ensure_dir, save_json


def compute_metrics(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute standard binary classification metrics."""
    y_pred = model.predict(x_test)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
    return metrics


def save_evaluation_artifacts(
    metrics: dict[str, Any],
    model_comparison: pd.DataFrame,
    model: Any,
    artifact_dir: Path | str = "artifacts",
) -> None:
    """Save all required deliverables into artifacts/."""
    base = Path(artifact_dir)
    ensure_dir(base)
    ensure_dir(base / "figures")

    save_json(metrics, base / "metrics.json")
    model_comparison.to_csv(base / "model_comparison.csv", index=False)
    joblib.dump(model, base / "best_model.joblib")

    # TODO: add richer visuals (ROC, calibration, error slices) in final submission.
    _try_save_confusion_matrix_figure(
        cm=np.asarray(metrics.get("confusion_matrix")),
        out_path=base / "figures" / "confusion_matrix.png",
    )


def _try_save_confusion_matrix_figure(cm: np.ndarray, out_path: Path) -> None:
    """Best-effort save confusion matrix figure if matplotlib is available."""
    if cm.size == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_saved_model(
    model_path: Path | str = "artifacts/best_model.joblib",
    data_path: Path | str = "data/raw/Telecom_customer churn.csv",
    artifact_dir: Path | str = "artifacts",
    target_col: str = "churn",
    random_state: int = 42,
) -> dict[str, Any]:
    """Load persisted model and recompute metrics on held-out split."""
    model = joblib.load(model_path)
    normalize = getattr(data_io, "standardize_column_names", lambda d: d)
    df = normalize(data_io.load_raw_data(data_path))
    x, y = data_io.split_features_target(df, target_col=target_col)
    x = prepare_feature_frame(x)
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    metrics = compute_metrics(model=model, x_test=x_test, y_test=y_test)
    save_json(metrics, Path(artifact_dir) / "metrics.json")
    _try_save_confusion_matrix_figure(
        cm=np.asarray(metrics.get("confusion_matrix")),
        out_path=Path(artifact_dir) / "figures" / "confusion_matrix.png",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved churn model.")
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.joblib")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/Telecom_customer churn.csv",
    )
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_saved_model(
        model_path=args.model_path,
        data_path=args.data_path,
        artifact_dir=args.artifact_dir,
    )
