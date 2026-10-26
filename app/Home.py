from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"

st.set_page_config(page_title="Heavy Equipment Price Prediction", page_icon="🚜", layout="wide")

st.title("Heavy Equipment Price Prediction")
st.markdown("Predicting the resale price of used heavy machinery with a stacked ensemble of "
            "XGBoost, LightGBM and CatBoost.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training sales", "138,701")
c2.metric("Test sales", "15,000")
c3.metric("Stacked model, 5-fold CV RMSLE", "0.19677")
c4.metric("Leaderboard RMSLE", "0.19099")

st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("The problem")
    st.markdown(
        "Each row is one completed sale of a machine such as a track excavator, motor grader, "
        "wheel loader, track-type tractor or backhoe loader. A row holds the machine's "
        "specification codes, manufacture year, hour-meter reading, sale date, region and up to "
        "30 component attributes.\n\n"
        "**Goal:** predict `TargetValue`, the sale price in US dollars, for every sale in the test set."
    )

    st.subheader("How it is scored")
    st.latex(r"\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(\log(1+\hat{y}_i) - \log(1+y_i)\big)^2}")
    st.markdown(
        "RMSLE measures **relative** error: missing by \\$5,000 on a \\$20,000 machine counts more than "
        "on a \\$120,000 one. Lower is better.\n\n"
        "Because RMSLE is just RMSE on `log(1 + price)`, every model is trained on `log1p(price)` and "
        "its predictions are turned back into dollars with `expm1`."
    )

with right:
    st.subheader("Why train on the log of the price")
    st.image(str(ASSETS / "target_distribution.png"), width="stretch")
    st.markdown(
        "Prices run from \\$7,500 to \\$142,000 with a long right tail (skewness 0.97). With squared "
        "error on raw dollars, the few most expensive machines would dominate training. After "
        "`log1p` the distribution is close to normal (skewness −0.20), and every machine counts "
        "in proportion to its price."
    )

st.divider()
st.subheader("What is in this app")
st.markdown(
    "| Page | What it shows |\n"
    "|---|---|\n"
    "| **Data** | What the dataset contains and why it is hard to model |\n"
    "| **Pipeline** | Every step from the raw CSV to the final price, as a flow diagram |\n"
    "| **Models** | The three models, their scores, tuning and stacking |\n"
    "| **Predict** | Pick a machine, change its year, hours or sale date, and get a live price |"
)
