"""Model training entrypoint for churn prediction coursework."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src import data_io
from src.evaluate import compute_metrics, save_evaluation_artifacts
from src.preprocess import build_preprocessor, prepare_feature_frame
from src.utils import DEFAULT_RANDOM_STATE, set_global_seed, setup_logger


@dataclass
class TrainResult:
    """Container for train output."""

    best_model_name: str
    best_pipeline: Pipeline
    metrics: dict[str, Any]
    comparison_df: pd.DataFrame


def _fallback_standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback standardization used if older module state lacks helper."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
    )
    return out


def build_model_registry(random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, Any]:
    """Return candidate models for benchmarking."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_and_select(
    data_path: Path | str,
    artifact_dir: Path | str = "artifacts",
    target_col: str = "churn",
    model_names: list[str] | None = None,
    drop_high_missing: bool = False,
    missing_threshold: float = 0.40,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TrainResult:
    """Train candidate models and persist required output artifacts."""
    logger = setup_logger()
    set_global_seed(random_state)

    # (1) Problem framing: define binary churn objective and data split protocol.
    normalize = getattr(data_io, "standardize_column_names", _fallback_standardize_column_names)
    df = normalize(data_io.load_raw_data(data_path))
    x, y = data_io.split_features_target(df, target_col=target_col)
    x = prepare_feature_frame(x)
    missing_rate = x.isna().mean()
    high_missing_cols = (
        missing_rate[missing_rate > missing_threshold].sort_values(ascending=False).index.tolist()
    )
    if drop_high_missing and high_missing_cols:
        x = x.drop(columns=high_missing_cols)
        logger.info(
            "Dropped %s columns above missing threshold %.2f: %s",
            len(high_missing_cols),
            missing_threshold,
            high_missing_cols,
        )

    # (2) EDA: TODO add richer profile reports and leakage checks in notebook/report.
    logger.info("Dataset loaded with %s rows and %s columns.", x.shape[0], x.shape[1])

    # (3) Preparation: split once and keep held-out test set untouched for final metrics.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    preprocessor = build_preprocessor(x_train)

    # (4) Modelling: evaluate multiple models on CV with a shared preprocessing pipeline.
    registry = build_model_registry(random_state=random_state)
    if model_names is not None:
        registry = {k: v for k, v in registry.items() if k in model_names}
    if not registry:
        raise ValueError("No models selected for training.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    best_name = ""
    best_cv_score = float("-inf")
    best_pipe: Pipeline | None = None

    for name, estimator in registry.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        scores = cross_validate(
            pipe,
            x_train,
            y_train,
            cv=cv,
            scoring={"pr_auc": "average_precision", "f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=None,
        )
        mean_pr_auc = float(scores["test_pr_auc"].mean())
        mean_f1 = float(scores["test_f1"].mean())
        mean_roc_auc = float(scores["test_roc_auc"].mean())
        rows.append(
            {
                "model": name,
                "drop_high_missing": drop_high_missing,
                "missing_threshold": missing_threshold,
                "dropped_cols_count": len(high_missing_cols) if drop_high_missing else 0,
                "cv_pr_auc_mean": mean_pr_auc,
                "cv_pr_auc_std": float(scores["test_pr_auc"].std()),
                "cv_f1_mean": mean_f1,
                "cv_f1_std": float(scores["test_f1"].std()),
                "cv_roc_auc_mean": mean_roc_auc,
                "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            }
        )
        logger.info(
            "Model=%s CV PR-AUC=%.4f | F1=%.4f | ROC-AUC=%.4f",
            name,
            mean_pr_auc,
            mean_f1,
            mean_roc_auc,
        )

        if mean_pr_auc > best_cv_score:
            best_cv_score = mean_pr_auc
            best_name = name
            best_pipe = pipe

    assert best_pipe is not None
    comparison_df = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)

    # (5) Fine-tune & Evaluate: TODO replace with search space tuning in final iteration.
    best_pipe.fit(x_train, y_train)
    metrics = compute_metrics(best_pipe, x_test, y_test)
    metrics["selected_model"] = best_name
    metrics["selection_metric"] = "cv_pr_auc_mean"
    metrics["selection_value"] = best_cv_score
    metrics["drop_high_missing"] = drop_high_missing
    metrics["missing_threshold"] = missing_threshold
    metrics["dropped_columns"] = high_missing_cols if drop_high_missing else []

    # (6) Final solution: persist chosen model and summary artifacts for reproducibility.
    save_evaluation_artifacts(metrics, comparison_df, best_pipe, artifact_dir=artifact_dir)

    return TrainResult(
        best_model_name=best_name,
        best_pipeline=best_pipe,
        metrics=metrics,
        comparison_df=comparison_df,
    )


def parse_args() -> argparse.Namespace:
    """CLI argument parser for training script."""
    parser = argparse.ArgumentParser(description="Train churn model and save artifacts.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/Telecom_customer churn.csv",
        help="Path to raw Telco churn CSV.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        help="Directory for model and metric outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model whitelist (e.g. logistic_regression random_forest).",
    )
    parser.add_argument(
        "--drop-high-missing",
        action="store_true",
        help="Drop columns above --missing-threshold before preprocessing.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.40,
        help="Missingness threshold used with --drop-high-missing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_select(
        data_path=args.data_path,
        artifact_dir=args.artifact_dir,
        model_names=args.models,
        drop_high_missing=args.drop_high_missing,
        missing_threshold=args.missing_threshold,
    )
