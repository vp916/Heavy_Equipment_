import math
from pathlib import Path

import pandas as pd
import streamlit as st

from common import UNKNOWN_YEARS, artifacts_ready, fmt, get_bundle, load_train
from pipeline import preprocess

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

st.set_page_config(page_title="Data · Heavy Equipment", page_icon="🚜", layout="wide")
st.title("The data")

overview, browse = st.tabs(["Overview and challenges", "Browse the data"])

with overview:

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Columns", "50", help="49 input columns plus the target, TargetValue (the sale price in dollars).")
    c2.metric("Missing cells", "47.9%", help="Share of all cells in train.csv that are empty. Another 534,061 cells hold the text \"None or Unspecified\", a disguised missing value.")
    c3.metric("Complete rows", "0", help="Not a single sale has every column filled in, so dropping incomplete rows is not an option.")
    c4.metric("Model codes", "3,505", help="Distinct values of Spec_FullDescriptor, the full model name of a machine, e.g. 320CL (excavator), 950FII (wheel loader), D6R (tractor).")

    st.subheader("What each row contains")
    st.dataframe(pd.DataFrame([
        ["Identifiers", "TransactionID, AssetID, ProductConfigID", "AssetID is nearly unique (120,111 values)"],
        ["Source and partner", "DataOriginCode, VendorPartnerID", "6 and 31 values"],
        ["Time", "TransactionDate, ManufactureYear", "Sales from 1990 to 2013"],
        ["Usage", "OperationalHoursMeter, UtilizationTier", "40.8% and 63.6% missing"],
        ["Specification", "Spec_FullDescriptor, Spec_BaseClass, Spec_SubClass, Spec_ReleaseSeries, "
                          "Spec_VariantModifier", "3,505 / 1,249 / 147 / 113 / 132 values"],
        ["Classification", "FunctionalClassification, InventoryGroupDescription, AssetScaleFactor",
         "e.g. “Wheel Loader - 150.0 to 175.0 Horsepower”"],
        ["Location", "RegionCode", "53 regions"],
        ["Components", "CabinType, DrivetrainType, Forks, col1–col30", "Anonymised; described only briefly in metadata.csv"],
    ], columns=["Group", "Columns", "Notes"]), hide_index=True, width="stretch")

    st.divider()
    st.header("Why this data is hard")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("1. Placeholder years")
        st.markdown("14,719 rows (10.6%) have `ManufactureYear = 1001`, which means *unknown*. Used as-is, "
                    "those machines would look about 1,000 years old. The pipeline turns it into a "
                    "missing value plus a `ManufactureYearMissing` flag.")
        st.image(str(ASSETS / "manufacture_year_distribution.png"), width="stretch")
    with right:
        st.subheader("2. Structured missingness")
        st.markdown("Component columns are filled in for one machine type only, so a missing value usually "
                    "means *not applicable to this machine*. The pipeline keeps it as its own category, "
                    "`\"Not Applicable\"`.")
        st.image(str(ASSETS / "missing_data_map.png"), width="stretch")
        st.dataframe(pd.DataFrame([
            ["Track Excavators", "71,018", "col20–col25"],
            ["Motor Graders", "23,405", "col6–col9, col11, col13, col14"],
            ["Wheel Loader", "18,872", "col29, col30"],
            ["Track Type Tractors", "14,959", "col27, col28"],
            ["Backhoe Loaders", "10,381", "col4, col5"],
        ], columns=["Machine type", "Rows", "Its own component columns"]), hide_index=True, width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("3. A long tail of rare model codes")
        st.markdown("Half of the 3,505 model codes appear five times or fewer and 776 appear only once. "
                    "80 model codes in the test set never appear in training. This is why the pipeline "
                    "uses **smoothed** target encoding: rare codes are pulled toward the overall average "
                    "instead of trusting a handful of sales.")
        st.subheader("4. Missing values in disguise")
        st.markdown("534,061 cells hold the text `\"None or Unspecified\"`. 40.8% of hour readings are missing "
                    "and another 22.8% are exactly 0, and the remaining readings include impossible values "
                    "(maximum 1,729,600 hours, about 197 years of non-stop use).")
    with right:
        st.subheader("5. A hard noise floor")
        st.markdown("Even identical-looking machines sell at different prices:")
        st.dataframe(pd.DataFrame([
            ["Nothing (whole dataset)", 0.671],
            ["Model code", 0.327],
            ["Model code + manufacture year", 0.300],
            ["Model code + manufacture year + sale year + region", 0.144],
        ], columns=["Machines grouped by", "Spread of log price within a group"]),
            hide_index=True, width="stretch")
        st.markdown("A spread of 0.14 is about ±15% for otherwise identical machines, and 6,792 rows are exact "
                    "duplicates of another row that often sold at a different price. No model can explain that "
                    "part, which is why the final RMSLE stays around 0.19.")
        st.subheader("6. A changing market")
        st.markdown("The median price moves with the economy: \\$31,000 in 1990, \\$43,000 in 1992, "
                    "\\$32,500 in 2003, \\$40,000 in 2007, \\$31,500 in 2010. Sale-date features let the "
                    "models separate the market trend from depreciation.")


PAGE_SIZE = 20
KEY_RAW = ["TransactionID", "TargetValue", "TransactionDate", "ManufactureYear", "OperationalHoursMeter",
           "InventoryGroupDescription", "Spec_FullDescriptor", "Spec_BaseClass", "FunctionalClassification",
           "RegionCode", "CabinType", "DrivetrainType", "UtilizationTier"]
KEY_ENGINEERED = ["TransactionYear", "TransactionMonth", "ManufactureYear", "ManufactureYearMissing", "AssetAge",
                  "OperationalHoursMeter", "OperationalHoursLog", "HoursPerYear", "HoursMissing",
                  "FC_MachineFamily", "FC_LowerBound", "FC_UpperBound", "FC_Unit", "Spec_FullDescriptor",
                  "Spec_BaseClass", "RegionCode"]


def go_to(page):
    st.session_state.browse_page = page


with browse:
    train = load_train()
    if train is None:
        st.info("Put train.csv in the project folder to browse the data.")
        st.stop()

    st.markdown("Every sale in `train.csv`. The filters search all 138,701 rows, and the table shows the "
                f"matching sales {PAGE_SIZE} at a time. Click a column header to sort the current page.")

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        machine = f1.selectbox("Machine type", ["All"] + sorted(train["InventoryGroupDescription"].dropna().unique()))
        region = f2.selectbox("Region", ["All"] + sorted(train["RegionCode"].dropna().unique()))
        code_query = f3.text_input("Model code contains", placeholder="e.g. 320C or D6",
                                   help="Searches Spec_FullDescriptor, the full model code. Not case-sensitive.")

        g1, g2, g3 = st.columns(3)
        real_years = train.loc[~train["ManufactureYear"].isin(UNKNOWN_YEARS), "ManufactureYear"]
        year_range = g1.slider("Manufacture year", int(real_years.min()), int(real_years.max()),
                               (int(real_years.min()), int(real_years.max())))
        include_unknown = g1.checkbox("Include unknown years (1001)", value=True,
                                      help="14,719 sales have the placeholder year 1001, which means unknown.")
        price_range = g2.slider("Sale price ($)", int(train["TargetValue"].min()), int(train["TargetValue"].max()),
                                (int(train["TargetValue"].min()), int(train["TargetValue"].max())), step=500)
        sort_by = g3.selectbox("Sort by", ["Original order", "Sale price", "Manufacture year", "Sale date"])
        descending = g3.toggle("Highest / newest first", value=True, disabled=sort_by == "Original order")

    view = st.radio("View", ["Raw data (as in train.csv)", "After feature engineering (what the models see)"],
                    horizontal=True,
                    help="Raw: the original columns. After feature engineering: the same sales after cleaning, new "
                         "features and ordinal encoding, i.e. the 61 numbers the models are trained on.")
    show_all = st.toggle("Show all columns", value=False,
                         help="Off: the most important columns only. On: every column (scroll sideways).")

    mask = pd.Series(True, index=train.index)
    if machine != "All":
        mask &= train["InventoryGroupDescription"] == machine
    if region != "All":
        mask &= train["RegionCode"] == region
    if code_query.strip():
        mask &= train["Spec_FullDescriptor"].str.contains(code_query.strip(), case=False, regex=False, na=False)
    unknown_year = train["ManufactureYear"].isin(UNKNOWN_YEARS)
    mask &= train["ManufactureYear"].between(*year_range) | (unknown_year & include_unknown)
    mask &= train["TargetValue"].between(*price_range)
    rows = train[mask]

    sort_column = {"Sale price": "TargetValue", "Manufacture year": "ManufactureYear",
                   "Sale date": "TransactionDate"}.get(sort_by)
    if sort_column:
        rows = rows.sort_values(sort_column, ascending=not descending, kind="stable")

    # back to page 1 whenever the filters change
    filters = (machine, region, code_query, year_range, include_unknown, price_range, sort_by, descending)
    if st.session_state.get("browse_filters") != filters:
        st.session_state.browse_filters = filters
        st.session_state.browse_page = 1

    n_pages = max(1, math.ceil(len(rows) / PAGE_SIZE))
    page = min(st.session_state.get("browse_page", 1), n_pages)
    start = (page - 1) * PAGE_SIZE
    page_rows = rows.iloc[start:start + PAGE_SIZE]

    if rows.empty:
        st.warning("No sales match these filters.")
        st.stop()

    b1, b2, info, b3, b4 = st.columns([1, 1, 3, 1, 1], vertical_alignment="center")
    b1.button("⏮ First", on_click=go_to, args=(1,), disabled=page == 1, width="stretch")
    b2.button("◀ Previous", on_click=go_to, args=(page - 1,), disabled=page == 1, width="stretch")
    info.markdown(f"<div style='text-align:center'><b>{len(rows):,}</b> matching sales · rows "
                  f"{start + 1:,}–{start + len(page_rows):,} · page <b>{page:,}</b> of {n_pages:,}</div>",
                  unsafe_allow_html=True)
    b3.button("Next ▶", on_click=go_to, args=(page + 1,), disabled=page == n_pages, width="stretch")
    b4.button("Last ⏭", on_click=go_to, args=(n_pages,), disabled=page == n_pages, width="stretch")

    if view.startswith("Raw"):
        table = page_rows if show_all else page_rows[KEY_RAW]
        st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table),
                     column_config={"TargetValue": st.column_config.NumberColumn("TargetValue ($)", format="%d"),
                                    "TransactionID": st.column_config.NumberColumn(format="%d")})
        st.caption("Empty cells are missing values. Note the year 1001 (unknown), the many empty component "
                   "columns and values like \"None or Unspecified\".")
    elif not artifacts_ready():
        st.info("This view needs the saved pipeline. From the project root, run `python app/export_artifacts.py`.")
    else:
        bundle = get_bundle()
        engineered = preprocess(page_rows.drop(columns="TargetValue"), bundle["fill_cols"],
                                bundle["machine_cols"], bundle["transformer"])[bundle["feature_cols"]]
        table = engineered if show_all else engineered[KEY_ENGINEERED]
        table = pd.concat([page_rows[["TransactionID", "TargetValue"]], table.round(4)], axis=1)
        st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table),
                     column_config={"TargetValue": st.column_config.NumberColumn("TargetValue ($)", format="%d"),
                                    "TransactionID": st.column_config.NumberColumn(format="%d")})
        st.caption("The same sales after feature engineering. Text columns are now integer codes (-1 = missing or "
                   "unseen), missing numbers are filled with the training median, and new columns such as AssetAge "
                   "and HoursPerYear are added. XGBoost and LightGBM additionally replace 8 high-cardinality codes "
                   "with their smoothed average log price inside each fold.")
