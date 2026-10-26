from pathlib import Path

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

st.set_page_config(page_title="Models · Heavy Equipment", page_icon="🚜", layout="wide")
st.title("The models")
st.markdown("Three gradient-boosted tree models, each predicting `log1p(price)`, combined by a Ridge regression.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("XGBoost CV", "0.20060", help="5-fold cross-validated RMSLE of XGBoost on its own (baseline settings kept after tuning).")
c2.metric("LightGBM CV (tuned)", "0.20025", help="5-fold cross-validated RMSLE of LightGBM with the tuned settings. The best single model.")
c3.metric("CatBoost CV", "0.20214", help="5-fold cross-validated RMSLE of CatBoost on its own. The weakest alone, but the most useful in the stack.")
c4.metric("Ridge stack CV", "0.19677", delta="-0.00348 vs best single", delta_color="inverse", help="The three models combined by a Ridge regression. Green because lower RMSLE is better.")

st.subheader("How boosting works")
st.markdown("All three start from the average log price and then add thousands of small trees. Each new tree "
            "is fitted to the **residuals**, the errors that are still left, and added with a small learning "
            "rate, so every tree fixes only a few percent of the remaining error.")
st.latex(r"F_m(x) = F_{m-1}(x) + \eta \cdot f_m(x)")

st.subheader("The three models")
st.dataframe(pd.DataFrame([
    ["XGBoost", "Leaf by leaf, up to 255 leaves", "Ordinal + target-encoded",
     "lr 0.02, min child weight 20, subsample 0.82, colsample 0.5, λ 2.5, α 0.15", "Baseline kept", "0.20060"],
    ["LightGBM", "Leaf by leaf, up to 383 leaves", "Ordinal + target-encoded",
     "lr 0.02, min child samples 10, subsample 0.75, colsample 0.4, λ 6", "Tuned", "0.20025"],
    ["CatBoost", "Symmetric trees, depth 8", "Ordinal codes",
     "lr 0.05, L2 5, up to 20,000 trees", "Baseline kept", "0.20214"],
], columns=["Model", "Tree growth", "Input", "Settings", "Tuning", "5-fold CV RMSLE"]),
    hide_index=True, width="stretch")
st.caption("All three use early stopping: training stops once the validation fold has not improved for 300 trees.")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Score in every fold")
    st.image(str(ASSETS / "validation_rmsle_across_folds.png"), width="stretch")
with right:
    st.subheader("Hyperparameter tuning")
    st.dataframe(pd.DataFrame([
        ["XGBoost", 0.20060, 0.20074, "Baseline kept"],
        ["LightGBM", 0.20041, 0.20032, "Tuned settings used"],
        ["CatBoost", 0.20214, 0.20219, "Baseline kept"],
    ], columns=["Model", "Baseline CV", "Best search score", "Decision"]),
        hide_index=True, width="stretch", column_config={
            "Baseline CV": st.column_config.NumberColumn(format="%.5f"),
            "Best search score": st.column_config.NumberColumn(format="%.5f")})
    st.markdown("The search is leak-free (the encoder sits inside the search's `Pipeline`), so its scores are "
                "honest. It found little to gain: the big improvements came from fixing the model settings "
                "themselves, such as column sampling 0.5, bigger trees and letting CatBoost train to convergence.")

st.divider()
st.header("Stacking")
left, right = st.columns([2, 3], gap="large")
with left:
    st.markdown("A Ridge regression learns one weight per model from their out-of-fold predictions:")
    st.dataframe(pd.DataFrame([
        ["XGBoost", 0.249, "24.8%"],
        ["LightGBM", 0.327, "32.5%"],
        ["CatBoost", 0.430, "42.8%"],
    ], columns=["Model", "Ridge weight", "Share"]), hide_index=True, width="stretch")
    st.markdown("CatBoost has the weakest single score but the **largest** weight. Its symmetric trees on "
                "ordinal codes make the most *different* mistakes, and a stack gains most from a model whose "
                "errors the others do not share. XGBoost and LightGBM see the same inputs and grow trees the "
                "same way, so their errors overlap (correlation 0.99).")
with right:
    st.image(str(ASSETS / "experiment_progression.png"), width="stretch")
