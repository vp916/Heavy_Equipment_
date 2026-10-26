import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import UNKNOWN_YEARS, artifacts_ready, get_bundle, model_code_stats, money, signed_money
from pipeline import predict

st.set_page_config(page_title="Predict · Heavy Equipment", page_icon="🚜", layout="wide")
st.title("Predict a sale price")

if not artifacts_ready():
    st.error("The trained models are not saved yet. From the project root, run:\n\n"
             "`python app/export_artifacts.py`\n\n(about 8 minutes, only needed once)")
    st.stop()

bundle = get_bundle()
templates, counts = bundle["templates"], bundle["model_counts"]
options = bundle["options"]
stats = model_code_stats()
UNKNOWN = "Unknown"
STACK_RMSLE = 0.19677   # 5-fold OOF score of the Ridge stack
LINE = "#2a78d6"

st.markdown("Describe a machine and its sale, and the final models predict what it would sell for. The three "
            "models were refitted on all 138,701 training sales with the settings chosen in the notebook. Your input "
            "goes through the same cleaning, feature engineering and encoding as the training data, and the Ridge "
            "stack combines the three predictions.")

mode = st.radio("Input", ["Build a machine", "Real sale from the test set"], horizontal=True,
                help="Build a machine: pick a model and change its details yourself. "
                     "Real sale: price one of 300 actual sales from the test set, exactly as they appear in test.csv.")


def pick(label, values, default, key, help=None):
    values = [UNKNOWN] + list(values)
    index = values.index(default) if default in values else 0
    choice = st.selectbox(label, values, index=index, key=key, help=help)
    return np.nan if choice == UNKNOWN else choice


def known_year(value):
    return None if value in UNKNOWN_YEARS else int(value)


if mode == "Build a machine":
    machine_box, usage_box, sale_box = st.columns(3, gap="medium")

    with machine_box.container(border=True):
        st.markdown("**Machine**")
        groups = bundle["group_of"].value_counts().index.tolist()
        group = st.selectbox("Machine type", groups,
                             help="The five main machine families in the data (plus a few skid steer loaders).")
        codes = counts[bundle["group_of"] == group]
        code = st.selectbox("Model code (most sold first)", codes.index.tolist(),
                            format_func=lambda c: f"{c}  ·  {counts[c]:,} sales",
                            help="Spec_FullDescriptor, the full model name, e.g. 320CL. Type to search. The number "
                                 "is how many sales of this model are in the training data.")

        row = templates.loc[[code]].reset_index(drop=True)
        t = row.iloc[0]

        year_unknown = st.checkbox("Manufacture year unknown", value=known_year(t["ManufactureYear"]) is None,
                                   key=f"year_unknown_{code}",
                                   help="10.6% of sales have no real year (stored as 1001). The models learned "
                                        "to handle that case.")
        year = st.number_input("Manufacture year", 1950, 2013, known_year(t["ManufactureYear"]) or 2000,
                               disabled=year_unknown, key=f"year_{code}",
                               help="Together with the sale date this gives the machine's age, one of the "
                                    "strongest price drivers.")
        cabin = pick("Cabin type", options["CabinType"], t["CabinType"], f"cabin_{code}",
                     help="EROPS: enclosed cab (Enclosed Roll-Over Protective Structure). EROPS w AC: enclosed cab "
                          "with air conditioning. OROPS: open canopy (Open Roll-Over Protective Structure).")
        drive = pick("Drivetrain", options["DrivetrainType"], t["DrivetrainType"], f"drive_{code}",
                     help="Transmission type, e.g. Standard, Powershift or Hydrostatic. Empty for most sales.")
        st.caption("Every other specification field is filled with the most common value for this model code "
                   "in the training data.")

    with usage_box.container(border=True):
        st.markdown("**Usage**")
        hours_known = pd.notna(t["OperationalHoursMeter"])
        hours_unknown = st.checkbox("Hour-meter reading unknown", value=not hours_known, key=f"hours_unknown_{code}",
                                    help="40.8% of sales have no reading and 22.8% show exactly 0, so only about a "
                                         "third of machines have a usable figure.")
        hours = st.number_input("Hour-meter reading", 0, 100_000,
                                int(t["OperationalHoursMeter"]) if hours_known else 3000, step=500,
                                disabled=hours_unknown, key=f"hours_{code}",
                                help="Total hours the engine has run. The models use it as log(hours) and as hours "
                                     "per year of age.")
        tier = pick("Utilisation tier", options["UtilizationTier"], t["UtilizationTier"], f"tier_{code}",
                    help="Low / Medium / High: how heavily the machine was used compared with similar machines. "
                         "Missing for 64% of sales.")

    with sale_box.container(border=True):
        st.markdown("**Sale**")
        sale_date = st.date_input("Sale date", dt.date(2012, 6, 1),
                                  min_value=dt.date(1990, 1, 1), max_value=dt.date(2013, 12, 31),
                                  help="The training data covers sales from 1990 to 2013. Prices follow the market, "
                                       "so the same machine is worth different amounts in different years.")
        region = pick("Region", options["RegionCode"], t["RegionCode"], f"region_{code}",
                      help="US state where the sale took place (53 regions).")

    row.loc[0, "ManufactureYear"] = 1001 if year_unknown else year
    row.loc[0, "OperationalHoursMeter"] = np.nan if hours_unknown else hours
    row.loc[0, "TransactionDate"] = sale_date.isoformat()
    row.loc[0, ["RegionCode", "UtilizationTier", "CabinType", "DrivetrainType"]] = [region, tier, cabin, drive]

else:
    sample = bundle["test_sample"]
    labels = {i: (f"{r.TransactionID}  ·  {r.Spec_FullDescriptor}  ·  "
                  f"{'unknown year' if r.ManufactureYear in UNKNOWN_YEARS else int(r.ManufactureYear)}  ·  "
                  f"{r.RegionCode}  ·  sold {r.TransactionDate}")
              for i, r in sample.iterrows()}
    i = st.selectbox("Test-set sale", list(labels), format_func=labels.get,
                     help="300 random sales from test.csv: TransactionID · model code · manufacture year · "
                          "region · sale date. Their true prices are hidden on the leaderboard.")
    row = sample.loc[[i], bundle["raw_cols"]].reset_index(drop=True)

with st.expander("Every input field sent to the model"):
    st.caption("All 49 raw columns of this sale, exactly as the pipeline receives them. Empty cells show as None/nan.")
    st.dataframe(row.T.rename(columns={0: "value"}).astype(str), width="stretch", height=400)

result = predict(row[bundle["raw_cols"]], bundle).iloc[0]
low, high = np.expm1(result["stack"] - STACK_RMSLE), np.expm1(result["stack"] + STACK_RMSLE)

# the same machine, sold on the same date, built in different years (used in the explanation and the chart)
sale_day = pd.to_datetime(row.loc[0, "TransactionDate"])
years = np.arange(max(1970, sale_day.year - 35), sale_day.year + 1)
curve_rows = pd.concat([row[bundle["raw_cols"]]] * len(years), ignore_index=True)
curve_rows["ManufactureYear"] = years
curve = pd.DataFrame({"Manufacture year": years, "Predicted price": predict(curve_rows, bundle)["price"].values})
curve_price = dict(zip(curve["Manufacture year"], curve["Predicted price"]))

st.divider()
left, right = st.columns([1, 2], gap="large")
with left:
    with st.container(border=True):
        st.metric("Predicted sale price", f"${result['price']:,.0f}",
                  help="The Ridge stack of XGBoost, LightGBM and CatBoost, converted back from log price to dollars.")
        st.markdown(f"Typical range: **{money(low)} – {money(high)}**")
        st.caption(f"± one RMSLE ({STACK_RMSLE:.3f} in log space, about ±20%). Part of this spread can never be "
                   "removed: identical machines in the data sold at different prices.")
with right:
    weights = bundle["ridge"]["coef"]
    st.markdown("**What each model predicts**")
    st.dataframe(pd.DataFrame({
        "Model": ["XGBoost", "LightGBM", "CatBoost"],
        "Predicted price": [f"${np.expm1(result[m]):,.0f}" for m in ("xgb", "lgb", "cat")],
        "Ridge weight": [weights[m] for m in ("xgb", "lgb", "cat")],
    }), hide_index=True, width="stretch",
        column_config={"Ridge weight": st.column_config.NumberColumn(
            format="%.3f", help="How much the stack trusts each model, learned from out-of-fold predictions.")})
    st.caption(f"The stack works on log prices: log price = {bundle['ridge']['intercept']:+.4f} + "
               + " + ".join(f"{weights[m]:.3f} × {m.upper()}" for m in ("xgb", "lgb", "cat"))
               + ". When the three models agree closely, the prediction is more trustworthy.")

# Why this price
code = row.loc[0, "Spec_FullDescriptor"]
chosen = known_year(row.loc[0, "ManufactureYear"])
hours_value = row.loc[0, "OperationalHoursMeter"]
reasons = []

if chosen:
    age = sale_day.year - chosen
    reasons.append(f"**Age.** Built in {chosen} and sold in {sale_day:%B %Y}, so the machine was **{age} years old** "
                   "at the sale. Age is one of the strongest price drivers in the data.")
    newer, older = curve_price.get(chosen + 1), curve_price.get(chosen - 1)
    if newer is not None and older is not None:
        reasons.append(f"**One year makes a difference.** Built a year later it would be predicted at {money(newer)} "
                       f"({signed_money(newer - result['price'])}), a year earlier at {money(older)} "
                       f"({signed_money(older - result['price'])}). See the curve below.")
else:
    reasons.append("**Age unknown.** The manufacture year is unknown, as for 10.6% of sales, so the models lean more "
                   "on the model code and the other details.")

if pd.notna(hours_value) and hours_value > 0:
    reasons.append(f"**Usage.** The hour meter shows {hours_value:,.0f} hours.")
else:
    reasons.append("**Usage unknown.** There is no usable hour-meter reading (true for about two thirds of sales), "
                   "so usage adds little information here.")

history = None
if stats is not None and code in stats.index:
    history = stats.loc[code]
    diff = result["price"] / history["median"] - 1
    reasons.append(f"**This model code.** {code} appears in {int(history['sales']):,} past sales with a median price "
                   f"of {money(history['median'])}. This prediction is **{abs(diff):.0%} "
                   f"{'above' if diff >= 0 else 'below'}** that median, because of this machine's age, sale date, "
                   "region and other details.")
    if history["sales"] <= 5:
        reasons.append("**Rare model code.** With so few past sales, the models rely more on broader information "
                       "such as the base class and the size class. This is exactly what the smoothed target "
                       "encoding is for.")
elif code not in counts.index:
    reasons.append(f"**Unseen model code.** {code} never appears in the training data, so the models fall back on "
                   "broader information such as the base class and the size class.")

st.subheader("Why this price?")
why, hist_col = st.columns([3, 2], gap="large")
with why:
    st.markdown("\n".join(f"- {r}" for r in reasons))
with hist_col:
    if history is not None:
        with st.container(border=True):
            st.markdown(f"**{code} in the training data**")
            first, last = history["first_year"], history["last_year"]
            built = "unknown" if pd.isna(first) else (f"{int(first)}" if first == last else f"{int(first)} – {int(last)}")
            st.dataframe(pd.DataFrame({
                "": ["Past sales", "Median price", "Middle half of prices", "Built"],
                "Value": [f"{int(history['sales']):,}", f"${history['median']:,.0f}",
                          f"${history['p25']:,.0f} – ${history['p75']:,.0f}", built],
            }), hide_index=True, width="stretch")
            st.caption("Half of all past sales of this model sold within the middle-half range.")
    elif stats is None:
        st.caption("Put train.csv in the project folder to see this model code's past sales.")

# Depreciation curve
st.subheader("How the price changes with manufacture year")
st.caption(f"The same machine, sold on {sale_day:%d %B %Y}, with only the manufacture year changed. Hover over the "
           "line to read the prices. The curve is almost flat for very old machines: few of them are in the data, "
           "and tree models cannot extrapolate beyond what they have seen."
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
