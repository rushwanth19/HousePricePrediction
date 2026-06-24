from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
TARGET_LABEL = "Median house value ($100,000s)"

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data…")
def load_data():
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.rename("MedHouseValue")
    return X, y


# --------------------------------------------------------------------------- #
# 2. Model (preprocessing + RandomForest in one pipeline, trained once)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training model (one-time)…")
def train_model(_X: pd.DataFrame, _y: pd.Series):
    numeric = _X.select_dtypes(include="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
    ])

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=RANDOM_STATE)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
    }
    return pipe, metrics, (X_test, y_test, preds)


# --------------------------------------------------------------------------- #
# 3. Load everything
# --------------------------------------------------------------------------- #
try:
    X, y = load_data()
except Exception as exc:
    st.error(
        "Couldn't load the California Housing dataset (it downloads via scikit-learn on "
        f"first use and needs internet).\n\nDetails: {exc}"
    )
    st.stop()

model, metrics, (X_test, y_test, test_preds) = train_model(X, y)
feature_cols = list(X.columns)


# --------------------------------------------------------------------------- #
# 4. Header + metrics
# --------------------------------------------------------------------------- #
st.title("🏠 House Price Prediction")
st.caption("Adjust the house features in the sidebar to get a live price estimate. "
           "Model: RandomForest trained on California Housing data.")

c1, c2, c3 = st.columns(3)
c1.metric("Model R²", f"{metrics['R2']:.3f}")
c2.metric("RMSE", f"{metrics['RMSE']:.3f}")
c3.metric("MAE", f"{metrics['MAE']:.3f}")


# --------------------------------------------------------------------------- #
# 5. Sidebar inputs (built from the data's own min/median/max)
# --------------------------------------------------------------------------- #
NICE = {
    "MedInc": "Median income (block, $10k)",
    "HouseAge": "House age (years)",
    "AveRooms": "Average rooms",
    "AveBedrms": "Average bedrooms",
    "Population": "Block population",
    "AveOccup": "Average occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

st.sidebar.header("Describe the house")
st.sidebar.caption("Sliders default to the dataset median.")

user_input = {}
for col in feature_cols:
    lo, hi, mid = float(X[col].min()), float(X[col].max()), float(X[col].median())
    step = (hi - lo) / 100 or 0.01
    user_input[col] = st.sidebar.slider(NICE.get(col, col), lo, hi, mid, step=step)

input_df = pd.DataFrame([user_input])[feature_cols]


# --------------------------------------------------------------------------- #
# 6. Tabs: prediction / performance / data
# --------------------------------------------------------------------------- #
tab_predict, tab_perf, tab_data = st.tabs(
    ["🎯 Prediction", "📈 Model performance", "🔎 Explore data"])

with tab_predict:
    prediction = float(model.predict(input_df)[0])
    st.subheader("Estimated value")
    st.metric(TARGET_LABEL, f"{prediction:.2f}", help="In units of $100,000")
    st.success(f"≈ **${prediction * 100_000:,.0f}**")

    with st.expander("Inputs sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)

    rf = model.named_steps["model"]
    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.feature_importances_, index=names).sort_values()
    st.subheader("What drives the prediction")
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.tail(10).plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Importance"); ax.set_title("Top feature importances")
    fig.tight_layout()
    st.pyplot(fig)

with tab_perf:
    st.subheader("Predicted vs actual (held-out test set)")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, test_preds, alpha=0.25, edgecolor="none")
    lims = [min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Residuals")
    residuals = y_test.values - test_preds
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(test_preds, residuals, alpha=0.25, edgecolor="none")
    ax2.axhline(0, color="r", linestyle="--", linewidth=1)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Residual")
    fig2.tight_layout()
    st.pyplot(fig2)

with tab_data:
    st.subheader("Sample of the data")
    st.dataframe(X.assign(**{TARGET_LABEL: y}).head(200), use_container_width=True)
    st.subheader("Feature summary")
    st.dataframe(X.describe().T, use_container_width=True)
