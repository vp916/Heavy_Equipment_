"""Data loading and feature engineering, as in notebooks/model_development.ipynb."""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

import config


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Replaces each category with its mean log price, pulled toward the global mean for rare categories:
    (n * category_mean + alpha * global_mean) / (n + alpha). Unseen categories get the global mean."""

    def __init__(self, columns, alpha=15):
        self.columns = columns
        self.alpha = alpha
        self.global_mean = None
        self.category_means = {}

    def fit(self, X, y):
        self.global_mean = y.mean()

        for col in self.columns:
            grouped = pd.DataFrame({"target": y, "category": X[col]})
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


def parse_functional_classification(df):
    """Splits labels like "Wheel Loader - 150.0 to 175.0 Horsepower" into machine family, drive type,
    size range and unit."""
    if "FunctionalClassification" not in df.columns:
        return df

    data = df.copy()
    fc = data["FunctionalClassification"].fillna("Unidentified")

    # reindex: with expand=True a batch without any "-" or "," gets no second column (e.g. a single row);
    # on the full data the output is identical to the notebook
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


def clean_machine_measurements(df):
    """Turns the size columns stored as text (<12', 13", 16 inch, 10' 2") into numbers."""
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
        values = data["col21"].astype(str).str.replace(" inch", "", regex=False).str.strip()
        data["col21_Numeric_Inches"] = pd.to_numeric(values, errors="coerce")
        data.drop(columns="col21", inplace=True)

    if "col22" in data.columns:

        def convert_reach(value):
            if pd.isna(value):
                return np.nan

            value = str(value).strip()
            if value.lower() in {"none", "nan", "not applicable", "none or unspecified", ""}:
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


def feature_engineering(df, fill_cols, machine_cols):
    """Date parts, placeholder years, asset age, missing flags and the fills for missing categories."""
    df = df.copy()
    df.drop(columns=["TransactionID", "InventoryGroupCategory", "col18", "col19"], errors="ignore", inplace=True)

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

    df["TransactionYear"] = df["TransactionDate"].dt.year
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["TransactionQuarter"] = df["TransactionDate"].dt.quarter
    df["TransactionDay"] = df["TransactionDate"].dt.day
    df["TransactionDayOfWeek"] = df["TransactionDate"].dt.dayofweek

    df.drop(columns="TransactionDate", inplace=True)

    df["ManufactureYearMissing"] = df["ManufactureYear"].isin([1000, 1001]).astype(int)
    df["ManufactureYear"] = df["ManufactureYear"].replace([1000, 1001], np.nan)
    df["AssetAge"] = df["TransactionYear"] - df["ManufactureYear"]
    df.loc[df["AssetAge"] < 0, "AssetAge"] = np.nan
    df["HoursMissing"] = df["OperationalHoursMeter"].isna().astype(int)

    df["Missing_BaseClass"] = df["Spec_BaseClass"].isna().astype(int)
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in machine_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Not Applicable")

    return df


class DataTransformer:
    """Fitted on train only: median fill for numeric columns, log hours, hours per year,
    ordinal codes for every text column (unseen or missing -> -1)."""

    def __init__(self, num_strategy="median"):
        self.num_imputer = SimpleImputer(strategy=num_strategy)
        self.ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                              encoded_missing_value=-1)

    def fit(self, X):
        X = X.copy()
        self.num_cols = [
            col for col in X.select_dtypes(include="number").columns
            if col not in ["TargetValue", "TransactionID", "AssetID"] and X[col].isna().any()
        ]
        self.cat_cols = X.select_dtypes(include="object").columns.tolist()

        self.num_imputer.fit(X[self.num_cols])
        self.ordinal_encoder.fit(X[self.cat_cols].astype(str))
        return self

    def transform(self, X):
        X = X.copy()
        if self.num_cols:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])

        X["OperationalHoursLog"] = np.log1p(X["OperationalHoursMeter"])
        X["HoursPerYear"] = X["OperationalHoursMeter"] / (X["AssetAge"] + 1)

        if self.cat_cols:
            X[self.cat_cols] = self.ordinal_encoder.transform(X[self.cat_cols].astype(str))
            X[self.cat_cols] = X[self.cat_cols].astype(int)

        X.drop(columns=["TransactionID", "AssetID"], errors="ignore", inplace=True)
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def prepare_data():
    """Loads train.csv and test.csv and runs the full feature engineering.

    Returns X, y, y_log, X_test, test_ids and target_cols (the columns that get target encoding).
    """
    trainData = pd.read_csv(config.TRAIN_CSV, low_memory=False)
    testData = pd.read_csv(config.TEST_CSV, low_memory=False)

    test_ids = testData["TransactionID"].copy()

    # Text columns with more than 20% missing, meant to get an "Unknown" fill. With pandas 3 the text
    # columns have dtype `str`, so this check selects nothing; their missing values get code -1 later.
    missing_ratio = trainData.isnull().mean()
    fill_cols = missing_ratio[(missing_ratio > 0.20) & (trainData.dtypes == "object")].index.tolist()
    fill_cols = [c for c in fill_cols if not c.startswith("col")]

    # anonymised component columns: missing means "the machine has no such part"
    machine_cols = [c for c in trainData.select_dtypes(include="object").columns if c.startswith("col")]

    trainData = parse_functional_classification(trainData)
    trainData = clean_machine_measurements(trainData)
    trainData = feature_engineering(trainData, fill_cols, machine_cols)

    transformer = DataTransformer()
    trainData = transformer.fit_transform(trainData)

    testData = parse_functional_classification(testData)
    testData = clean_machine_measurements(testData)
    testData = feature_engineering(testData, fill_cols, machine_cols)
    testData = transformer.transform(testData)

    # categorical columns with more than 20 values get target encoding
    target_cols = [c for c in transformer.cat_cols if trainData[c].nunique() > 20]

    y = trainData["TargetValue"]
    X = trainData.drop(columns=["TargetValue"])
    X_test = testData.copy()
    y_log = np.log1p(y)

    target_cols = [c for c in target_cols if c in X.columns]

    return X, y, y_log, X_test, test_ids, target_cols
