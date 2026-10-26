from pathlib import Path

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

st.set_page_config(page_title="Data · Heavy Equipment", page_icon="🚜", layout="wide")
st.title("The data")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Columns", "50")
c2.metric("Missing cells", "47.9%")
c3.metric("Complete rows", "0")
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
