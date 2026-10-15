# Milestone versions

This folder holds the single-model notebooks that the project was built from. Each team member developed and tuned one model in their own notebook. The three models were then combined, retrained and stacked in the final pipeline, [`../model_development.ipynb`](../model_development.ipynb).

| Notebook | Version | What it does | Baseline CV RMSLE | Tuning search score | Retrained in final pipeline |
|---|---|---|---:|---:|---:|
| [`training_xgboost.ipynb`](training_xgboost.ipynb) | V1 | XGBoost on ordinal + in-fold target-encoded features; `RandomizedSearchCV` | 0.20161 | 0.19675 * | 0.20189 |
| [`training_lightgbm.ipynb`](training_lightgbm.ipynb) | V2 | LightGBM on ordinal + in-fold target-encoded features; `RandomizedSearchCV` | 0.20215 | 0.19800 * | 0.20169 |
| [`training_catboost.ipynb`](training_catboost.ipynb) | V3 | CatBoost on ordinal features; `RandomizedSearchCV` (baseline kept) | 0.20699 | 0.20769 | 0.20699 |
| [`../model_development.ipynb`](../model_development.ipynb) | Final | All three models + tuning + Ridge stacking → `submission.csv` | | | **0.19805** (stack) |

\* Optimistic: inside the tuning step, the target encoder is fitted on all training rows before `RandomizedSearchCV`. The in-fold retraining scores in the last column are the reliable ones.

All notebooks share the same EDA and feature-engineering cells. They read the data from the project root (`../../train.csv`, `../../test.csv`).

Apart from an added header cell, the notebooks are unchanged, with their original code and outputs.
