"""Build a preprocessing pipeline that handles numeric and categorical columns.

Using a ColumnTransformer keeps preprocessing inside the model pipeline, so the
exact same transformations are applied at train and predict time (no leakage).
"""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_column_types(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = split_column_types(X)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
