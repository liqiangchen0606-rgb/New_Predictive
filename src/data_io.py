"""Data IO helpers for Telco churn coursework pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils import ensure_dir


DEFAULT_DATA_PATH = Path("data/raw/Telecom_customer churn.csv")


def load_raw_data(csv_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load raw churn data from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. Download via Kaggle and place it there."
        )
    return pd.read_csv(path)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with standardized snake_case lowercase column names."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
    )
    return out


def resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    """Resolve target column with exact, case-insensitive, then common fallback names."""
    cols = df.columns.tolist()
    if target_col in cols:
        return target_col

    target_low = target_col.lower()
    lower_map = {c.lower(): c for c in cols}
    if target_low in lower_map:
        return lower_map[target_low]

    for candidate in ["churn", "target", "label", "is_churn"]:
        if candidate in lower_map:
            return lower_map[candidate]
    raise KeyError(f"Target column '{target_col}' not found.")


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Churn",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features X and binary target y."""
    resolved_target = resolve_target_column(df, target_col)

    x = df.drop(columns=[resolved_target]).copy()
    y = df[resolved_target].copy()

    # TODO: confirm all label variants in the chosen dataset and adjust mapping if needed.
    if y.dtype == "object":
        y = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target contains unmapped values after label conversion.")
    return x, y.astype(int)


def ensure_project_layout(base_dir: Path | str = ".") -> None:
    """Create expected repo folders for data and artifacts."""
    base = Path(base_dir)
    for rel in [
        "data/raw",
        "data/processed",
        "artifacts",
        "artifacts/figures",
        "notebooks",
        "tests/data",
    ]:
        ensure_dir(base / rel)
