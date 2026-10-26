"""Preprocessing and prediction for the Streamlit app.

The classes and functions below are copied from notebooks/model_development.ipynb
(cells 23, 24, 25, 33 and 34) so that a single row typed into the app is
transformed exactly like the training data.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


# ---------------------------------------------------------------- notebook cell 23
class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns, alpha=15):
        self.columns = columns
        self.alpha = alpha
        self.global_mean = None
        self.category_means = {}

    def fit(self, X, y):

        self.global_mean = y.mean()

        for col in self.columns:

            grouped = pd.DataFrame({
                "target": y,
                "category": X[col]
            })
            stats = grouped.groupby("category")["target"].agg(["mean", "count"])

            self.category_means[col] = (
                (stats["mean"] * stats["count"] + self.alpha * self.global_mean)
                / (stats["count"] + self.alpha)
            )
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:

            X[col] = X[col].map(self.category_means[col])
            X[col] = X[col].fillna(self.global_mean)

        return X


# ---------------------------------------------------------------- notebook cell 24
def parse_functional_classification(df):

    if "FunctionalClassification" not in df.columns:
        return df

    data = df.copy()

    fc = data["FunctionalClassification"].fillna("Unidentified")

    # reindex: with expand=True a single row without "-" or "," gets no second column
    # (the only change from the notebook; identical output on the full data)
    split_data = fc.str.split(r"\s*-\s*", n=1, expand=True).reindex(columns=[0, 1])

    machine = split_data[0].str.split(",", n=1, expand=True).reindex(columns=[0, 1])

    data["FC_MachineFamily"] = machine[0].str.strip()
    data["FC_DriveType"] = machine[1].fillna("None").str.strip()

    specs = split_data[1].fillna("Unidentified")

    lower = specs.str.extract(r"(\d+\.?\d*)")[0]
    upper = specs.str.extract(r"to\s*(\d+\.?\d*)")[0]
    unit = specs.str.extract(r"(Metric Tons|Horsepower|Ft[^,]*|Lb[^,]*)")[0]

    data["FC_LowerBound"] = lower.astype(float)
    data["FC_UpperBound"] = upper.astype(float)
    data["FC_Unit"] = unit.fillna("Unknown")

    data["FC_IsPlusRange"] = specs.str.contains(r"\+").astype(int)
    data["FC_RangeKnown"] = (~specs.str.contains("Unidentified")).astype(int)

    return data


# ---------------------------------------------------------------- notebook cell 25
def clean_machine_measurements(df):

    data = df.copy()

    if "col7" in data.columns:
        values = data["col7"].astype(str).str.replace(r"[<' ]", "", regex=True)
        data["col7_Numeric_Feet"] = pd.to_numeric(values, errors="coerce")
        data.drop(columns="col7", inplace=True)

    if "col15" in data.columns:
        values = data["col15"].astype(str).str.replace('"', "", regex=False)
        data["col15_Numeric_Inches"] = pd.to_numeric(values, errors="coerce")
        data.drop(columns="col15", inplace=True)

    if "col21" in data.columns:
        values = (
            data["col21"]
            .astype(str)
            .str.replace(" inch", "", regex=False)
            .str.strip()
        )

        data["col21_Numeric_Inches"] = pd.to_numeric(values, errors="coerce")
        data.drop(columns="col21", inplace=True)

    if "col22" in data.columns:

        def convert_reach(value):

            if pd.isna(value):
                return np.nan

            value = str(value).strip()

            if value.lower() in {
                "none",
                "nan",
                "not applicable",
                "none or unspecified",
                ""
            }:
                return np.nan

            match = re.match(r"(\d+)\s*'\s*(\d*)\s*\"?", value)
            if match:
                feet = float(match.group(1))
                inches = float(match.group(2) or 0)
                return feet * 12 + inches

            return np.nan

        data["col22_Total_Inches"] = data["col22"].apply(convert_reach)
        data.drop(columns="col22", inplace=True)

    return data


# ---------------------------------------------------------------- notebook cell 33
def feature_engineering(df, fill_cols, machine_cols):
    df = df.copy()
    df.drop(columns=["TransactionID", "InventoryGroupCategory", "col18", "col19"], errors="ignore", inplace=True)

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

    df["TransactionYear"] = df["TransactionDate"].dt.year
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["TransactionQuarter"] = df["TransactionDate"].dt.quarter
    df["TransactionDay"] = df["TransactionDate"].dt.day
    df["TransactionDayOfWeek"] = df["TransactionDate"].dt.dayofweek

    df.drop(columns="TransactionDate", inplace=True)

    df["ManufactureYearMissing"] = (df["ManufactureYear"].isin([1000, 1001]).astype(int))
    df["ManufactureYear"] = (df["ManufactureYear"].replace([1000, 1001], np.nan))
    df["AssetAge"] = (df["TransactionYear"] - df["ManufactureYear"])
    df.loc[df["AssetAge"] < 0, "AssetAge"] = np.nan
    df["HoursMissing"] = (df["OperationalHoursMeter"].isna().astype(int))

    df["Missing_BaseClass"] = (df["Spec_BaseClass"].isna().astype(int))
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in machine_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Not Applicable")

    return df


# ---------------------------------------------------------------- notebook cell 34
class DataTransformer:

    def __init__(self, num_strategy="median"):
        self.num_imputer = SimpleImputer(strategy=num_strategy)
        self.ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)

    def fit(self, X):
        X = X.copy()
        self.num_cols = [
            col
            for col in X.select_dtypes(include="number").columns
            if col not in ["TargetValue", "TransactionID", "AssetID"]
            and X[col].isna().any()]

        self.cat_cols = (X.select_dtypes(include="object").columns.tolist())
        self.num_imputer.fit(X[self.num_cols])

        self.ordinal_encoder.fit(X[self.cat_cols].astype(str))

        return self

    def transform(self, X):
        X = X.copy()
        if self.num_cols:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])

        X["OperationalHoursLog"] = np.log1p(X["OperationalHoursMeter"])
        X["HoursPerYear"] = (X["OperationalHoursMeter"] / (X["AssetAge"] + 1))

        if self.cat_cols:
            X[self.cat_cols] = self.ordinal_encoder.transform(X[self.cat_cols].astype(str))
            X[self.cat_cols] = X[self.cat_cols].astype(int)

        X.drop(columns=["TransactionID", "AssetID"], errors="ignore", inplace=True)

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------- app helpers
def preprocess(raw, fill_cols, machine_cols, transformer):
    """Raw rows (same columns as train.csv) -> model features, as in notebook cells 35-36."""
    data = parse_functional_classification(raw)
    data = clean_machine_measurements(data)
    data = feature_engineering(data, fill_cols, machine_cols)
    return transformer.transform(data)


def load_artifacts():
    """Load everything export_artifacts.py saved."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    bundle = joblib.load(ARTIFACTS / "pipeline.joblib")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(ARTIFACTS / "xgb.ubj")
    lgb_model = lgb.Booster(model_file=str(ARTIFACTS / "lgb.txt"))
    cat_model = CatBoostRegressor()
    cat_model.load_model(str(ARTIFACTS / "cat.cbm"))

    bundle["models"] = {"xgb": xgb_model, "lgb": lgb_model, "cat": cat_model}
    return bundle


def predict(raw, bundle):
    """Predict prices for raw rows. Returns one log-prediction per model, the stack, and the price."""
    raw = raw.astype(bundle["raw_dtypes"][raw.columns])
    X = preprocess(raw, bundle["fill_cols"], bundle["machine_cols"], bundle["transformer"])
    X = X[bundle["feature_cols"]]
    X_enc = bundle["encoder"].transform(X)

    models = bundle["models"]
    log_preds = pd.DataFrame({
        "xgb": models["xgb"].predict(X_enc),
        "lgb": models["lgb"].predict(X_enc),
        "cat": models["cat"].predict(X),
    }, index=raw.index)

    ridge = bundle["ridge"]
    stack = ridge["intercept"] + sum(ridge["coef"][name] * log_preds[name] for name in ridge["coef"])

    out = log_preds.copy()
    out["stack"] = stack
    out["price"] = np.maximum(np.expm1(stack), 0)
    return out
