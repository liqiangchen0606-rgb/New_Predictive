"""Preprocessing pipeline builders (leakage-safe with ColumnTransformer)."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# (1) Problem framing: define which columns are usable and remove ID-like fields.
def prepare_feature_frame(
    x: pd.DataFrame,
    drop_columns: Iterable[str] = ("customerID", "customerid", "customer_id"),
) -> pd.DataFrame:
    """Drop non-predictive identifier columns if present."""
    out = x.copy()
    to_drop = [c for c in drop_columns if c in out.columns]
    if to_drop:
        out = out.drop(columns=to_drop)
    return out


# (3) Preparation: build reproducible transforms using ColumnTransformer.
def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Return a preprocessing transformer for numeric + categorical columns."""
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            # Add missingness indicators so structural missingness can carry signal.
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
