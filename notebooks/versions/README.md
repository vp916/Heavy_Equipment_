# Milestone versions

This folder holds the single-model notebooks that the project was built from. Each team member developed and tuned one model in their own notebook. The three models were then combined, retrained and stacked in the final pipeline, [`../model_development.ipynb`](../model_development.ipynb).

| Notebook | Version | What it does | Baseline CV RMSLE | Tuning search score | Retrained in first final pipeline |
|---|---|---|---:|---:|---:|
| [`training_xgboost.ipynb`](training_xgboost.ipynb) | V1 | XGBoost on ordinal + in-fold target-encoded features; `RandomizedSearchCV` | 0.20161 | 0.19675  | 0.20189 |
| [`training_lightgbm.ipynb`](training_lightgbm.ipynb) | V2 | LightGBM on ordinal + in-fold target-encoded features; `RandomizedSearchCV` | 0.20215 | 0.19800  | 0.20169 |
| [`training_catboost.ipynb`](training_catboost.ipynb) | V3 | CatBoost on ordinal features; `RandomizedSearchCV` (baseline kept) | 0.20699 | 0.20769 | 0.20699 |
| [`../model_development.ipynb`](../model_development.ipynb) | Final | All three models + leak-free tuning + Ridge stacking → `submission.csv` | | | **0.19677** (stack, current version) |



The final pipeline has since been updated: new model settings and a leak-free tuning search (main README, Section 5.4). It improved the stack from 0.19805 to 0.19677 (leaderboard 0.19296 → 0.19099). The milestone notebooks are kept as they were, so their numbers refer to the first version.

All notebooks share the same EDA and feature-engineering cells. They read the data from the project root (`../../train.csv`, `../../test.csv`).


