"""Fit the final models on all training rows and save what the Streamlit app needs.

Run once from the project root:  python app/export_artifacts.py

Settings are the ones selected in notebooks/model_development.ipynb (LightGBM tuned,
XGBoost and CatBoost baseline). Without a validation fold there is no early stopping,
so each model gets a fixed tree count: the mean best iteration from the notebook's
5-fold run, scaled by 1.15 because the full data is 25% larger than a training fold.
"""
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

from pipeline import (ARTIFACTS, ROOT, DataTransformer, SmoothedTargetEncoder,
                      clean_machine_measurements, feature_engineering, load_artifacts,
                      parse_functional_classification, predict, preprocess)

DATA = ROOT.parent
ARTIFACTS.mkdir(exist_ok=True)

# best iterations in the notebook's final run: XGB 5467-6259 (mean 5785), LGB 3578-4989 (mean 4298),
# CatBoost about 14,000-19,000 (README section 5.4)
N_TREES = {"xgb": 6650, "lgb": 4950, "cat": 19000}

# Ridge meta-model from the notebook's final run (cell 61 output)
RIDGE = {"coef": {"xgb": 0.249, "lgb": 0.327, "cat": 0.430}, "intercept": -0.0651}

t0 = time.time()
trainData = pd.read_csv(DATA / "train.csv", low_memory=False)
testData = pd.read_csv(DATA / "test.csv", low_memory=False)
raw_cols = [c for c in trainData.columns if c != "TargetValue"]
raw_dtypes = trainData[raw_cols].dtypes

# notebook cells 27-31
missing_ratio = trainData.isnull().mean()
fill_cols = missing_ratio[(missing_ratio > 0.20) & (trainData.dtypes == "object")].index.tolist()
fill_cols = [c for c in fill_cols if not c.startswith("col")]
machine_cols = [c for c in trainData.select_dtypes(include="object").columns if c.startswith("col")]

# notebook cells 35-40
transformer = DataTransformer()
train_fe = feature_engineering(clean_machine_measurements(parse_functional_classification(trainData)),
                               fill_cols, machine_cols)
train_fe = transformer.fit_transform(train_fe)
target_cols = [c for c in transformer.cat_cols if train_fe[c].nunique() > 20]

y_log = np.log1p(train_fe["TargetValue"])
X = train_fe.drop(columns=["TargetValue"])
target_cols = [c for c in target_cols if c in X.columns]

encoder = SmoothedTargetEncoder(columns=target_cols, alpha=15)
X_enc = encoder.fit_transform(X, y_log)
print(f"data ready: {X.shape}, {len(target_cols)} target-encoded columns ({time.time() - t0:.0f}s)", flush=True)

models = {
    "xgb": xgb.XGBRegressor(
        n_estimators=N_TREES["xgb"], learning_rate=0.02, tree_method="hist", grow_policy="lossguide",
        max_leaves=255, max_depth=0, min_child_weight=20, subsample=0.82, colsample_bytree=0.5,
        reg_lambda=2.5, reg_alpha=0.15, random_state=42, n_jobs=-1),
    "lgb": lgb.LGBMRegressor(
        n_estimators=N_TREES["lgb"], learning_rate=0.02, subsample_freq=1, random_state=42, n_jobs=-1,
        verbose=-1, num_leaves=383, min_child_samples=10, subsample=0.75, colsample_bytree=0.4,
        reg_lambda=6.0),
    "cat": CatBoostRegressor(
        iterations=N_TREES["cat"], learning_rate=0.05, depth=8, l2_leaf_reg=5, loss_function="RMSE",
        random_seed=42, verbose=0),
}

for name, model in models.items():
    t = time.time()
    model.fit(X if name == "cat" else X_enc, y_log)
    print(f"{name} fitted on {len(X)} rows ({time.time() - t:.0f}s)", flush=True)

models["xgb"].save_model(ARTIFACTS / "xgb.ubj")
models["lgb"].booster_.save_model(str(ARTIFACTS / "lgb.txt"))
models["cat"].save_model(str(ARTIFACTS / "cat.cbm"))

# autofill template per model code: the most common value of every raw column among its sales
counts = trainData["Spec_FullDescriptor"].value_counts()
templates = {"Spec_FullDescriptor": pd.Series(counts.index, index=counts.index)}
for col in raw_cols:
    if col == "Spec_FullDescriptor":
        continue
    top = (trainData.groupby(["Spec_FullDescriptor", col], dropna=False).size()
           .reset_index(name="n").sort_values("n", ascending=False)
           .drop_duplicates("Spec_FullDescriptor").set_index("Spec_FullDescriptor")[col])
    templates[col] = top
templates = pd.DataFrame(templates).loc[counts.index]
templates = templates.astype(raw_dtypes)
group_of = templates["InventoryGroupDescription"]

options = {col: sorted(trainData[col].dropna().unique().tolist())
           for col in ["RegionCode", "UtilizationTier", "CabinType", "DrivetrainType"]}

test_sample = testData.sample(300, random_state=42).reset_index(drop=True)

joblib.dump({
    "raw_cols": raw_cols, "raw_dtypes": raw_dtypes, "fill_cols": fill_cols, "machine_cols": machine_cols,
    "transformer": transformer, "encoder": encoder, "target_cols": target_cols,
    "feature_cols": X.columns.tolist(), "ridge": RIDGE, "n_trees": N_TREES,
    "templates": templates, "model_counts": counts, "group_of": group_of, "options": options,
    "test_sample": test_sample,
}, ARTIFACTS / "pipeline.joblib")
print(f"artifacts saved ({time.time() - t0:.0f}s)", flush=True)

# Checks: reload the saved files exactly like the app does
bundle = load_artifacts()
pred = predict(testData[raw_cols], bundle)

# 1. the saved files reproduce the in-memory models
X_test = preprocess(testData, fill_cols, machine_cols, transformer)[X.columns]
direct = models["xgb"].predict(encoder.transform(X_test))
print(f"reload check, max |xgb diff|: {np.abs(direct - pred['xgb'].values).max():.2e}")

# 2. one row at a time gives the same answer as the whole test set at once
single = pd.concat([predict(testData[raw_cols].iloc[[i]], bundle) for i in range(50)])
print(f"single-row check, max |price diff|: ${np.abs(single['price'].values - pred['price'].values[:50]).max():.4f}")

# 3. close to the submitted predictions (5 fold models averaged vs one full-data model)
sub = pd.read_csv(DATA / "notebooks" / "submission.csv")
assert (sub["TransactionID"].values == testData["TransactionID"].values).all()
diff = np.log1p(pred["price"].values) - np.log1p(sub["TargetValue"].values)
report = {
    "rmsle_vs_submission": float(np.sqrt(np.mean(diff ** 2))),
    "mean_log_diff_vs_submission": float(diff.mean()),
    "corr_vs_submission": float(np.corrcoef(pred["price"], sub["TargetValue"])[0, 1]),
    "minutes": round((time.time() - t0) / 60, 1),
}
print(json.dumps(report, indent=2))
for p in sorted(ARTIFACTS.iterdir()):
    print(f"{p.name:16s} {p.stat().st_size / 1e6:8.1f} MB")
