import numpy as np
import pandas as pd
import streamlit as st

from common import UNKNOWN_YEARS, artifacts_ready, get_bundle, load_train, value_table
from pipeline import clean_machine_measurements, feature_engineering, parse_functional_classification, predict

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

# One sale through the pipeline

st.divider()
st.header("Follow one sale through the pipeline")
st.markdown("The steps above, applied to one real sale from `train.csv`. Each table shows only the columns that "
            "step changes or adds, so you can see exactly what happens to the data.")

train = load_train()
if train is None or not artifacts_ready():
    st.info("This walkthrough needs train.csv in the project folder and the saved models "
            "(`python app/export_artifacts.py`).")
    st.stop()

bundle = get_bundle()
complete = (train["FunctionalClassification"].notna() & ~train["ManufactureYear"].isin(UNKNOWN_YEARS)
            & train["OperationalHoursMeter"].gt(0))
# prefer sales that have at least one measurement value, so step 2 has something to show
has_measurement = pd.concat([train[c].notna() & (train[c] != "None or Unspecified")
                             for c in ["col7", "col15", "col21", "col22"]], axis=1).any(axis=1)
examples = {}
for kind in ["Track Excavators", "Wheel Loader", "Track Type Tractors", "Motor Graders", "Backhoe Loaders"]:
    candidates = train.index[complete & (train["InventoryGroupDescription"] == kind)]
    preferred = candidates[has_measurement[candidates].values]
    examples[kind] = preferred[0] if len(preferred) else candidates[0]
kind = st.selectbox("Example sale", list(examples),
                    format_func=lambda k: (f"{k}: {train.at[examples[k], 'Spec_FullDescriptor']}, built "
                                           f"{int(train.at[examples[k], 'ManufactureYear'])}, sold "
                                           f"{train.at[examples[k], 'TransactionDate']} in "
                                           f"{train.at[examples[k], 'RegionCode']}"))

raw = train.loc[[examples[kind]]]
x = raw[bundle["raw_cols"]].astype(bundle["raw_dtypes"])
step1 = parse_functional_classification(x)
step2 = clean_machine_measurements(step1)
step3 = feature_engineering(step2, bundle["fill_cols"], bundle["machine_cols"])
step4 = bundle["transformer"].transform(step3)[bundle["feature_cols"]]
step5 = bundle["encoder"].transform(step4)
result = predict(x, bundle).iloc[0]
r0, r1, r2, r3, r4, r5 = (df.iloc[0] for df in (raw, step1, step2, step3, step4, step5))


def show(title, text, rows, columns):
    st.subheader(title)
    st.markdown(text)
    st.dataframe(value_table(rows, columns), hide_index=True, width="stretch", height=38 + 35 * len(rows))


show("Step 0 · The raw sale", "A selection of the 49 raw columns, exactly as they appear in train.csv.",
     [[c, r0[c]] for c in ["TransactionDate", "ManufactureYear", "OperationalHoursMeter", "InventoryGroupDescription",
                           "Spec_FullDescriptor", "Spec_BaseClass", "FunctionalClassification", "RegionCode",
                           "CabinType", "DrivetrainType", "UtilizationTier"]]
     + [["TargetValue (the price to predict)", f"${r0['TargetValue']:,.0f}"]],
     ["Column", "Value"])

show("Step 1 · Parse the functional classification",
     f"The label *{r0['FunctionalClassification']}* is split into separate features.",
     [[c, r1[c]] for c in ["FC_MachineFamily", "FC_DriveType", "FC_LowerBound", "FC_UpperBound", "FC_Unit",
                           "FC_IsPlusRange", "FC_RangeKnown"]],
     ["New column", "Value"])

measured = [("col7", "col7_Numeric_Feet"), ("col15", "col15_Numeric_Inches"),
            ("col21", "col21_Numeric_Inches"), ("col22", "col22_Total_Inches")]
show("Step 2 · Clean the measurement columns",
     "Size columns stored as text become numbers. Columns that do not apply to this machine type stay empty "
     "for now and are filled with the median in step 4.",
     [[old, r0[old], new, r2[new]] for old, new in measured],
     ["Text column", "Raw value", "New column", "Number"])

component_cols = [c for c in bundle["machine_cols"] if c in r0.index]
filled = sum(pd.isna(r0[c]) for c in component_cols if c not in ("col18", "col19"))
show("Step 3 · Feature engineering",
     f"Date parts, the machine's age and missing-value flags. In addition, {filled} empty component columns of this "
     "machine are filled with \"Not Applicable\" because those parts do not exist on this machine type.",
     [[c, r3[c]] for c in ["TransactionYear", "TransactionMonth", "TransactionQuarter", "TransactionDay",
                           "TransactionDayOfWeek", "ManufactureYearMissing", "AssetAge", "HoursMissing",
                           "Missing_BaseClass"]],
     ["New column", "Value"])

median_filled = [c for c in bundle["transformer"].num_cols if pd.isna(r3[c])]
show("Step 4 · DataTransformer",
     "Missing numbers get the training median, two usage features are added, and every text column becomes an "
     "integer code. After this step the sale is 61 numbers.",
     [["OperationalHoursMeter", r3["OperationalHoursMeter"], r4["OperationalHoursMeter"]],
      ["OperationalHoursLog", "(new)", r4["OperationalHoursLog"]],
      ["HoursPerYear", "(new)", r4["HoursPerYear"]]]
     + [[f"{c} (median fill)", r3[c], r4[c]] for c in median_filled[:2]]
     + [[c, r3[c], r4[c]] for c in ["InventoryGroupDescription", "CabinType", "FC_MachineFamily", "RegionCode"]],
     ["Column", "Before", "After"])

show("Step 5 · Smoothed target encoding (XGBoost and LightGBM only)",
     "The 8 high-cardinality columns are replaced by their smoothed average log price in the training data. "
     "The last column turns that log price back into dollars to make it readable. CatBoost keeps the integer "
     "codes from step 4.",
     [[c, r3[c], r4[c], r5[c], f"${np.expm1(r5[c]):,.0f}"] for c in bundle["target_cols"]],
     ["Column", "Raw value", "Integer code", "Encoded (log price)", "As a price"])

st.subheader("Step 6 · Predict and stack")
weights = bundle["ridge"]["coef"]
st.markdown("Each model predicts the log price, the Ridge stack combines them with its learned weights, and "
            "`expm1` turns the result back into dollars.")
st.dataframe(value_table(
    [[name, f"{result[m]:.4f}", f"${np.expm1(result[m]):,.0f}", f"{weights[m]:.3f}"]
     for m, name in [("xgb", "XGBoost"), ("lgb", "LightGBM"), ("cat", "CatBoost")]]
    + [["Ridge stack", f"{result['stack']:.4f}", f"${result['price']:,.0f}", ""]],
    ["Model", "Predicted log price", "Predicted price", "Ridge weight"]), hide_index=True, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted price", f"${result['price']:,.0f}")
c2.metric("Actual sale price", f"${r0['TargetValue']:,.0f}")
c3.metric("Difference", f"{result['price'] / r0['TargetValue'] - 1:+.1%}",
          help="Relative difference between prediction and actual price.")
st.caption("This sale was part of the training data, and the app's models were trained on all training sales, so "
           "the prediction is closer to the actual price than it would be for a new machine. The honest estimate of "
           "the error on new sales is the cross-validation score, about ±20%.")
