import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import ARTIFACTS, load_artifacts, predict

st.set_page_config(page_title="Predict · Heavy Equipment", page_icon="🚜", layout="wide")
st.title("Predict a sale price")

if not (ARTIFACTS / "pipeline.joblib").exists():
    st.error("The trained models are not saved yet. From the project root, run:\n\n"
             "`python app/export_artifacts.py`")
    st.stop()


@st.cache_resource(show_spinner="Loading the trained models…")
def get_bundle():
    return load_artifacts()


bundle = get_bundle()
templates, counts = bundle["templates"], bundle["model_counts"]
options = bundle["options"]
UNKNOWN = "Unknown"
STACK_RMSLE = 0.19677   # 5-fold OOF score of the Ridge stack
LINE = "#2a78d6"

st.markdown("The three models were refitted on all 138,701 training sales with the settings chosen in the "
            "notebook. A row goes through the same cleaning, feature engineering and encoding as in training, "
            "then the Ridge stack combines the three predictions.")

mode = st.radio("Input", ["Build a machine", "Real sale from the test set"], horizontal=True)


def pick(label, values, default, key):
    values = [UNKNOWN] + list(values)
    index = values.index(default) if default in values else 0
    choice = st.selectbox(label, values, index=index, key=key)
    return np.nan if choice == UNKNOWN else choice


def known_year(value):
    return None if value in (1000, 1001) else int(value)


if mode == "Build a machine":
    machine_box, usage_box, sale_box = st.columns(3, gap="medium")

    with machine_box.container(border=True):
        st.markdown("**Machine**")
        groups = bundle["group_of"].value_counts().index.tolist()
        group = st.selectbox("Machine type", groups)
        codes = counts[bundle["group_of"] == group]
        code = st.selectbox("Model code (most sold first)", codes.index.tolist(),
                            format_func=lambda c: f"{c}  ·  {counts[c]:,} sales")

        row = templates.loc[[code]].reset_index(drop=True)
        t = row.iloc[0]

        year_unknown = st.checkbox("Manufacture year unknown", value=known_year(t["ManufactureYear"]) is None,
                                   key=f"year_unknown_{code}")
        year = st.number_input("Manufacture year", 1950, 2013, known_year(t["ManufactureYear"]) or 2000,
                               disabled=year_unknown, key=f"year_{code}")
        cabin = pick("Cabin type", options["CabinType"], t["CabinType"], f"cabin_{code}")
        drive = pick("Drivetrain", options["DrivetrainType"], t["DrivetrainType"], f"drive_{code}")
        st.caption("Every other specification field is filled with the most common value for this model code "
                   "in the training data.")

    with usage_box.container(border=True):
        st.markdown("**Usage**")
        hours_known = pd.notna(t["OperationalHoursMeter"])
        hours_unknown = st.checkbox("Hour-meter reading unknown", value=not hours_known, key=f"hours_unknown_{code}")
        hours = st.number_input("Hour-meter reading", 0, 100_000,
                                int(t["OperationalHoursMeter"]) if hours_known else 3000, step=500,
                                disabled=hours_unknown, key=f"hours_{code}")
        tier = pick("Utilisation tier", options["UtilizationTier"], t["UtilizationTier"], f"tier_{code}")

    with sale_box.container(border=True):
        st.markdown("**Sale**")
        sale_date = st.date_input("Sale date", dt.date(2012, 6, 1),
                                  min_value=dt.date(1990, 1, 1), max_value=dt.date(2013, 12, 31))
        region = pick("Region", options["RegionCode"], t["RegionCode"], f"region_{code}")

    row.loc[0, "ManufactureYear"] = 1001 if year_unknown else year
    row.loc[0, "OperationalHoursMeter"] = np.nan if hours_unknown else hours
    row.loc[0, "TransactionDate"] = sale_date.isoformat()
    row.loc[0, ["RegionCode", "UtilizationTier", "CabinType", "DrivetrainType"]] = [region, tier, cabin, drive]

else:
    sample = bundle["test_sample"]
    labels = {i: (f"{r.TransactionID}  ·  {r.Spec_FullDescriptor}  ·  "
                  f"{'unknown year' if r.ManufactureYear in (1000, 1001) else int(r.ManufactureYear)}  ·  "
                  f"{r.RegionCode}  ·  sold {r.TransactionDate}")
              for i, r in sample.iterrows()}
    i = st.selectbox("Test-set sale (300 random sales; the true price is hidden on the leaderboard)",
                     list(labels), format_func=labels.get)
    row = sample.loc[[i], bundle["raw_cols"]].reset_index(drop=True)

with st.expander("Every input field sent to the model"):
    st.dataframe(row.T.rename(columns={0: "value"}).astype(str), width="stretch", height=400)

result = predict(row[bundle["raw_cols"]], bundle).iloc[0]
low, high = np.expm1(result["stack"] - STACK_RMSLE), np.expm1(result["stack"] + STACK_RMSLE)

st.divider()
left, right = st.columns([1, 2], gap="large")
with left:
    with st.container(border=True):
        st.metric("Predicted sale price", f"${result['price']:,.0f}")
        st.markdown(f"Typical range: **\\${low:,.0f} – \\${high:,.0f}**")
        st.caption(f"± one RMSLE ({STACK_RMSLE:.3f} in log space, about ±20%). Part of this spread is "
                   "irreducible: identical machines in the data sold at different prices.")
with right:
    weights = bundle["ridge"]["coef"]
    st.dataframe(pd.DataFrame({
        "Model": ["XGBoost", "LightGBM", "CatBoost"],
        "Predicted price": [f"${np.expm1(result[m]):,.0f}" for m in ("xgb", "lgb", "cat")],
        "Ridge weight": [weights[m] for m in ("xgb", "lgb", "cat")],
    }), hide_index=True, width="stretch",
        column_config={"Ridge weight": st.column_config.NumberColumn(format="%.3f")})
    st.caption(f"Stack (log scale) = {bundle['ridge']['intercept']:+.4f} + "
               + " + ".join(f"{weights[m]:.3f} × {m.upper()}" for m in ("xgb", "lgb", "cat")))

# ---- depreciation curve: the same machine, sold on the same date, built in different years
sale_year = pd.to_datetime(row.loc[0, "TransactionDate"]).year
years = np.arange(max(1970, sale_year - 35), sale_year + 1)
curve_rows = pd.concat([row[bundle["raw_cols"]]] * len(years), ignore_index=True)
curve_rows["ManufactureYear"] = years
curve = pd.DataFrame({"Manufacture year": years, "Predicted price": predict(curve_rows, bundle)["price"].values})

st.subheader("How the price changes with manufacture year")
chosen = known_year(row.loc[0, "ManufactureYear"])
st.caption(f"The same machine, sold on {row.loc[0, 'TransactionDate']}, with only the manufacture year changed."
           + ("" if chosen else " This sale's manufacture year is unknown, so no year is highlighted."))

hover = alt.selection_point(fields=["Manufacture year"], nearest=True, on="pointerover",
                            clear="pointerout", empty=False)
base = alt.Chart(curve).encode(
    x=alt.X("Manufacture year:Q", scale=alt.Scale(domain=[int(years[0]), int(years[-1])], nice=False),
            axis=alt.Axis(format="d", tickMinStep=1, grid=False, title="Manufacture year")),
    y=alt.Y("Predicted price:Q", axis=alt.Axis(format="$,.0f", title=None, tickCount=5)),
)
tooltip = [alt.Tooltip("Manufacture year:Q", format="d"), alt.Tooltip("Predicted price:Q", format="$,.0f")]
layers = [
    base.mark_line(color=LINE, strokeWidth=2),
    base.mark_rule(color="#a8a59d", strokeWidth=1).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))).transform_filter(hover),
    base.mark_point(size=400, opacity=0).encode(tooltip=tooltip).add_params(hover),
    base.mark_point(filled=True, size=70, color=LINE, stroke="#fcfcfb", strokeWidth=2, opacity=1)
        .transform_filter(hover),
]
if chosen and years[0] <= chosen <= years[-1]:
    picked = curve[curve["Manufacture year"] == chosen]
    pt = alt.Chart(picked).encode(x="Manufacture year:Q", y="Predicted price:Q")
    layers += [
        pt.mark_point(filled=True, size=130, color=LINE, stroke="#fcfcfb", strokeWidth=2, opacity=1),
        pt.mark_text(align="right", dx=-10, dy=-14, fontWeight="bold", fontSize=13).encode(
            text=alt.value(f"{chosen}: ${picked['Predicted price'].iloc[0]:,.0f}")),
    ]
st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
