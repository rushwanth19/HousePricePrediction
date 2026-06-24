import streamlit as st

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Predict",
        "About"
    ]
)

if page == "Home":

    st.title("🏡 House Price Prediction")

    st.write("""
    This application predicts house prices
    using Machine Learning.
    """)

    st.metric("Best Model", "XGBoost")

    st.metric("R² Score", "91.4%")

elif page == "Predict":

    st.title("Prediction Page")

    st.info("Prediction form coming soon.")

elif page == "About":

    st.title("About")

    st.write("""
    Developed using

    - Python
    - Pandas
    - Scikit-Learn
    - XGBoost
    - Streamlit
    """)