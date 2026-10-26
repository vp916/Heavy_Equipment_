"""Helpers shared by the app pages: cached data and models, and small formatting functions."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import ARTIFACTS, load_artifacts

TRAIN_CSV = Path(__file__).resolve().parent.parent / "train.csv"
UNKNOWN_YEARS = (1000, 1001)


def artifacts_ready():
    return (ARTIFACTS / "pipeline.joblib").exists()


@st.cache_resource(show_spinner="Loading the trained models…")
def get_bundle():
    return load_artifacts()


@st.cache_resource
def get_importance():
    """Feature importance written by compute_importance.py, or None if it has not been run."""
    import joblib
    path = ARTIFACTS / "importance.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource(show_spinner="Reading train.csv…")
def load_train():
    """The raw training data, or None if train.csv is not in the project folder. Do not modify the result."""
    if not TRAIN_CSV.exists():
        return None
    return pd.read_csv(TRAIN_CSV, low_memory=False)


@st.cache_resource
def model_code_stats():
    """Sales count, price quartiles and year range per model code (Spec_FullDescriptor)."""
    train = load_train()
    if train is None:
        return None
    years = train["ManufactureYear"].where(~train["ManufactureYear"].isin(UNKNOWN_YEARS))
    groups = train.assign(Year=years).groupby("Spec_FullDescriptor")
    return pd.DataFrame({
        "sales": groups.size(),
        "median": groups["TargetValue"].median(),
        "p25": groups["TargetValue"].quantile(0.25),
        "p75": groups["TargetValue"].quantile(0.75),
        "first_year": groups["Year"].min(),
        "last_year": groups["Year"].max(),
    })


def money(value):
    """Dollar amount for st.markdown ($ is escaped so it is not read as LaTeX)."""
    return f"\\${value:,.0f}"


def signed_money(value):
    """Dollar difference with its sign, e.g. +$4,200 or −$3,100."""
    return f"{'+' if value >= 0 else '−'}\\${abs(value):,.0f}"


def fmt(value):
    """A cell value as readable text."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "missing"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def value_table(rows, columns):
    """Small table of already formatted values (all text, so mixed types display cleanly)."""
    return pd.DataFrame([[fmt(v) for v in row] for row in rows], columns=columns)


GROUP_DESCRIPTIONS = {
    "Model identity": "Which exact model the machine is: the full model code, its parts and the configuration ID",
    "Age": "How old the machine was when it was sold",
    "Machine type and size": "The machine family and its size class (e.g. 21 to 24 metric tons), parsed from the "
                             "functional classification",
    "Sale date": "When the sale happened, which captures the market level and the season",
    "Usage": "How much the machine has been used, from the hour meter",
    "Region": "The US state where it was sold",
    "Seller and data source": "Who sold it and where the record came from",
    "Components (col*, cab, drivetrain)": "Equipment details: cab, drivetrain, forks and the anonymised component "
                                          "columns, including the parsed measurements (sizes in feet or inches)",
}


def group_contents(importance):
    """One row per feature group: its share of the model's decisions, a description and all its columns."""
    members = {}
    for feature, group in importance["group_of"].items():
        members.setdefault(group, []).append(feature)
    shares = importance["groups"] / importance["groups"].sum()
    return pd.DataFrame({
        "Group": shares.index,
        "Share": [f"{s:.1%}" for s in shares.values],
        "What it means": [GROUP_DESCRIPTIONS.get(g, "") for g in shares.index],
        "Columns": [", ".join(members[g]) for g in shares.index],
    })
