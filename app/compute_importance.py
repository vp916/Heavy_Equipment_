"""Feature importance of the final stacked model with SHAP values. No retraining: it reads the saved models.

Run once from the project root, after export_artifacts.py:  python app/compute_importance.py

For a sample of training sales, each model's SHAP values split its predicted log price into one contribution per
feature (they add up exactly to the prediction). The three models' values are combined with the Ridge weights, so
they explain the stacked prediction. Writes app/artifacts/importance.joblib and assets/feature_importance.png.
"""
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline import ARTIFACTS, ROOT, load_artifacts, preprocess, stack_shap  # noqa: E402

N_ROWS = 2000
ASSETS = ROOT.parent / "assets"

FEATURE_GROUPS = {
    "Age": ["ManufactureYear", "AssetAge", "ManufactureYearMissing"],
    "Model identity": ["Spec_FullDescriptor", "Spec_BaseClass", "Spec_SubClass", "Spec_ReleaseSeries",
                       "Spec_VariantModifier", "ProductConfigID", "Missing_BaseClass"],
    "Machine type and size": ["InventoryGroupDescription", "FunctionalClassification", "AssetScaleFactor",
                              "FC_MachineFamily", "FC_DriveType", "FC_LowerBound", "FC_UpperBound", "FC_Unit",
                              "FC_IsPlusRange", "FC_RangeKnown"],
    "Sale date": ["TransactionYear", "TransactionMonth", "TransactionQuarter", "TransactionDay",
                  "TransactionDayOfWeek"],
    "Usage": ["OperationalHoursMeter", "OperationalHoursLog", "HoursPerYear", "HoursMissing", "UtilizationTier"],
    "Region": ["RegionCode"],
    "Seller and data source": ["VendorPartnerID", "DataOriginCode"],
}

t0 = time.time()
bundle = load_artifacts()
features = bundle["feature_cols"]
group_of = {f: group for group, members in FEATURE_GROUPS.items() for f in members}
group_of.update({f: "Components (col*, cab, drivetrain)" for f in features if f not in group_of})

train = pd.read_csv(ROOT.parent / "train.csv", low_memory=False)
sample = train.sample(N_ROWS, random_state=42)
X = preprocess(sample[bundle["raw_cols"]].astype(bundle["raw_dtypes"]), bundle["fill_cols"],
               bundle["machine_cols"], bundle["transformer"])[features]

contributions, base = stack_shap(X, bundle)          # (rows, features) in log price, plus the base value
print(f"SHAP values for {N_ROWS} sales ({time.time() - t0:.0f}s)", flush=True)

importance = pd.Series(np.abs(contributions).mean(axis=0), index=features).sort_values(ascending=False)
grouped = (pd.DataFrame(contributions, columns=features).T.groupby(group_of).sum().T
           .abs().mean().sort_values(ascending=False))

joblib.dump({"features": importance, "groups": grouped, "group_of": group_of, "n_rows": N_ROWS,
             "base_log_price": base}, ARTIFACTS / "importance.joblib")

print("\nTop 15 features (mean |SHAP|, log price):")
print(importance.head(15).round(4).to_string())
print("\nGroups:")
print((grouped / grouped.sum() * 100).round(1).astype(str).add(" %").to_string())

# figure for the README
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1, 1.25]})
fig.patch.set_facecolor("#fcfcfb")
for ax, data, title in [(axes[0], grouped / grouped.sum() * 100, "Share of the model's decisions per feature group"),
                        (axes[1], importance.head(15), "Top 15 features (mean |SHAP value|, log price)")]:
    data = data.iloc[::-1]
    ax.barh(data.index, data.values, color="#2a78d6", height=0.62)
    ax.set_facecolor("#fcfcfb")
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color="#1d1d1b", pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(axis="y", length=0, labelsize=10.5, colors="#3a3935")
    ax.tick_params(axis="x", labelsize=9.5, colors="#5f5d57")
    ax.xaxis.grid(True, color="#e3e3e3")
    ax.set_axisbelow(True)
    for y, v in enumerate(data.values):
        ax.text(v, y, f" {v:.1f}%" if ax is axes[0] else f" {v:.3f}", va="center", fontsize=9.5, color="#3a3935")
fig.text(0.01, 0.01, f"SHAP values of the stacked model (Ridge-weighted XGBoost, LightGBM and CatBoost) on "
         f"{N_ROWS:,} random training sales.", fontsize=9, color="#5f5d57")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(ASSETS / "feature_importance.png", dpi=110, facecolor=fig.get_facecolor())
print(f"\nSaved {ARTIFACTS / 'importance.joblib'} and {ASSETS / 'feature_importance.png'} "
      f"({(time.time() - t0) / 60:.1f} min)")
