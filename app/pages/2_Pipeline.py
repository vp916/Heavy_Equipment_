import streamlit as st

st.set_page_config(page_title="Pipeline · Heavy Equipment", page_icon="🚜", layout="wide")
st.title("The pipeline")
st.markdown("From the raw CSV files to a predicted price. Every step runs in "
            "`notebooks/model_development.ipynb`.")

st.graphviz_chart("""
digraph {
    rankdir=TB; nodesep=0.35; ranksep=0.4;
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, fillcolor="#eef3fb", color="#9db4d8"];
    edge [color="#7a8aa0"];

    raw   [label="train.csv / test.csv\\n138,701 + 15,000 sales, 50 columns", fillcolor="#f2f2f2", color="#bbbbbb"];
    fc    [label="1. Parse functional classification\\nmachine family, drive type, capacity range"];
    meas  [label="2. Clean measurements\\ncol7 / col15 / col21 / col22 → numbers"];
    fe    [label="3. Feature engineering\\ndate parts, asset age, missing flags, \\"Not Applicable\\""];
    dt    [label="4. DataTransformer\\nmedian fill, log hours, hours per year, ordinal codes"];
    log   [label="5. Target: log1p(price)", fillcolor="#fff6e5", color="#e0b860"];

    subgraph cluster_cv {
        label="6. 5-fold cross-validation (each fold)"; fontname="Helvetica"; fontsize=12;
        style="rounded,dashed"; color="#9db4d8";
        te  [label="Smoothed target encoder\\nfitted on the training fold only", fillcolor="#fff6e5", color="#e0b860"];
        xgb [label="XGBoost\\nordinal + target-encoded"];
        lgb [label="LightGBM\\nordinal + target-encoded"];
        cat [label="CatBoost\\nordinal codes"];
    }

    tune  [label="7. Hyperparameter tuning\\nRandomizedSearchCV, encoder inside the Pipeline"];
    ridge [label="8. Ridge stacking\\nlearns how much to trust each model", fillcolor="#e6f4ea", color="#7cbf8e"];
    out   [label="9. expm1 → predicted price\\nsubmission.csv · leaderboard 0.19099", fillcolor="#e6f4ea", color="#7cbf8e"];

    raw -> fc -> meas -> fe -> dt -> log -> te;
    te -> xgb; te -> lgb; log -> cat;
    xgb -> tune; lgb -> tune; cat -> tune;
    tune -> ridge -> out;
}
""", width="stretch")

st.header("Each step")

with st.expander("1. Parse the functional classification", expanded=False):
    st.markdown("`FunctionalClassification` is a text label such as *“Wheel Loader - 150.0 to 175.0 Horsepower”*. "
                "It is split into the machine family, the drive type and a numeric capacity range, so the "
                "models get an ordered size value instead of a free-text label.")
    st.code('data["FC_LowerBound"] = specs.str.extract(r"(\\d+\\.?\\d*)")[0].astype(float)\n'
            'data["FC_UpperBound"] = specs.str.extract(r"to\\s*(\\d+\\.?\\d*)")[0].astype(float)', language="python")

with st.expander("2. Clean measurement strings"):
    st.markdown("Some component columns hold measurements in mixed formats: `<12'`, `13\"`, `16 inch`, `10' 2\"`. "
                "They are converted to plain numbers (feet or inches) so trees can split on physical size.")
    st.code("10' 2\"  →  10 × 12 + 2 = 122 inches", language="text")

with st.expander("3. Feature engineering"):
    st.markdown(
        "- **Sale date** → year, month, quarter, day, day of week (market level and seasonality).\n"
        "- **Placeholder year 1001** → missing value plus a `ManufactureYearMissing` flag.\n"
        "- **`AssetAge`** = sale year − manufacture year (depreciation drives price).\n"
        "- **Missing flags** for hours and base class.\n"
        "- **Missing component values** (`col*`) → `\"Not Applicable\"`. Other missing text values get "
        "their own code (−1) in the next step.\n"
        "- **Dropped:** `TransactionID`, `AssetID` (nearly unique), `InventoryGroupCategory` (duplicate), "
        "`col18` / `col19` (99.95% empty)."
    )
    st.code('df["AssetAge"] = df["TransactionYear"] - df["ManufactureYear"]', language="python")

with st.expander("4. DataTransformer"):
    st.markdown("Fills remaining numeric gaps with the median, adds `OperationalHoursLog = log1p(hours)` and "
                "`HoursPerYear = hours / (age + 1)`, and turns every text column into an integer code. "
                "Categories never seen in training get −1.")

with st.expander("5. Log target"):
    st.markdown("Models are trained on `log1p(price)`. Squared error on this scale is exactly RMSLE, and it "
                "stops the most expensive machines from dominating training.")

with st.expander("6. 5-fold cross-validation and smoothed target encoding"):
    st.markdown("The training data is split into 5 folds. Each model trains on 4 folds and predicts the 5th, "
                "so every sale gets an **out-of-fold** prediction from a model that never saw it.")
    st.markdown("The 8 categorical columns with more than 20 levels (model code, base class, region, …) are "
                "replaced by their smoothed average log price:")
    st.latex(r"\text{TE}(c) = \frac{n_c\,\bar{y}_c + 15\,\bar{y}}{n_c + 15}")
    st.markdown("A code seen twice is pulled most of the way to the overall average; a code seen 1,000 times "
                "keeps its own average. The encoder is **fitted on the training folds only**, so the validation "
                "fold's prices never leak into its own features.")

with st.expander("7. Hyperparameter tuning"):
    st.markdown("`RandomizedSearchCV` tries 5 settings per model around the current ones. The target encoder "
                "and the model are wrapped in one sklearn `Pipeline`, so the encoder is refitted inside every "
                "search fold too. A tuned setting is kept only if it beats the model's baseline CV score.")

with st.expander("8. Ridge stacking"):
    st.markdown("A Ridge regression takes the three models' out-of-fold predictions as its inputs and learns "
                "how much weight each deserves. Because the models make partly different mistakes, the "
                "weighted blend beats every single model (0.19677 vs 0.20025 for the best one).")

with st.expander("9. Final prediction"):
    st.markdown("Test predictions are averaged over the 5 fold models, combined by the Ridge, converted back "
                "with `expm1` and clipped at 0, then written to `submission.csv`.")
