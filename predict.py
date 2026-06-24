"""Load the saved pipeline and score new rows.

Example:
    python predict.py            # scores a couple of demo rows
"""
from __future__ import annotations
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing

from config import MODELS_DIR


def load_model(path=None):
    path = path or (MODELS_DIR / "best_model.joblib")
    return joblib.load(path)


def predict(df: pd.DataFrame):
    model = load_model()
    return model.predict(df)


if __name__ == "__main__":
    # Demo: take a few rows from the bundled dataset and predict on them.
    sample = fetch_california_housing(as_frame=True).data.head(3)
    preds = predict(sample)
    for i, p in enumerate(preds):
        print(f"Row {i}: predicted value = {p:,.3f}")
