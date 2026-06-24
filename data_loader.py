"""Load data for the project.

Two sources are supported:
  - "california": the bundled scikit-learn dataset (default, no download).
  - "kaggle": the Ames/Kaggle `train.csv` placed in data/ (target = SalePrice).
"""
from __future__ import annotations
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data(dataset: str = "california", data_path: str | None = None):
    """Return (X, y) as a DataFrame and Series."""
    if dataset == "california":
        bunch = fetch_california_housing(as_frame=True)
        X = bunch.data.copy()
        y = bunch.target.rename("MedHouseValue")  # in $100,000s
        return X, y

    if dataset == "kaggle":
        if not data_path:
            raise ValueError("Pass --data-path pointing to the Kaggle train.csv")
        df = pd.read_csv(data_path)
        if "SalePrice" not in df.columns:
            raise ValueError("Expected a 'SalePrice' column in the Kaggle dataset.")
        y = df["SalePrice"]
        X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
        return X, y

    raise ValueError(f"Unknown dataset: {dataset!r}")
