"""Train and compare several regression models, then persist the best one."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS
from preprocessing import build_preprocessor


def candidate_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def train_and_select(X: pd.DataFrame, y: pd.Series):
    preprocessor = build_preprocessor(X)
    results = {}
    best_name, best_pipe, best_score = None, None, -np.inf

    for name, model in candidate_models().items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        # Negative RMSE so that "higher is better" for selection.
        scores = cross_val_score(
            pipe, X, y, cv=CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        mean_score = scores.mean()
        results[name] = -mean_score  # report positive RMSE
        print(f"{name:18s}  CV RMSE = {-mean_score:,.4f}  (+/- {scores.std():.4f})")

        if mean_score > best_score:
            best_name, best_score, best_pipe = name, mean_score, pipe

    print(f"\nBest model: {best_name} (CV RMSE = {-best_score:,.4f})")
    best_pipe.fit(X, y)  # refit best pipeline on all training data

    out = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipe, out)
    print(f"Saved fitted pipeline -> {out}")
    return best_name, best_pipe, results
