"""Evaluate a fitted pipeline on a held-out test set and save diagnostic plots."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # headless backend so it works on any machine / CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import REPORTS_DIR


def evaluate(pipe, X_test, y_test) -> dict:
    preds = pipe.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
    }
    print("\nHold-out test performance")
    for k, v in metrics.items():
        print(f"  {k:5s} = {v:,.4f}")

    _plot_pred_vs_actual(y_test, preds)
    _plot_residuals(y_test, preds)
    return metrics


def _plot_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pred_vs_actual.png", dpi=120)
    plt.close(fig)


def _plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.3, edgecolor="none")
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual plot")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "residuals.png", dpi=120)
    plt.close(fig)
