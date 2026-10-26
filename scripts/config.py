"""Paths and every setting used by the pipeline, in one place.

The values are the ones in notebooks/model_development.ipynb.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42
N_FOLDS = 5
TARGET_ENCODING_ALPHA = 15
EARLY_STOPPING_ROUNDS = 300
RIDGE_ALPHA = 1.0


# Settings that are never tuned: learning rate, tree limit, early stopping, seed.
FIXED_PARAMS = {
    "xgb": dict(n_estimators=15000, learning_rate=0.02, tree_method="hist", grow_policy="lossguide",
                max_depth=0, early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=SEED, n_jobs=-1),
    "lgb": dict(n_estimators=12000, learning_rate=0.02, subsample_freq=1, random_state=SEED, n_jobs=-1,
                verbose=-1),
    "cat": dict(iterations=20000, learning_rate=0.05, loss_function="RMSE", random_seed=SEED,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0),
}

# Baseline values of the settings that tuning may change.
BASELINE_PARAMS = {
    "xgb": dict(max_leaves=255, min_child_weight=20, subsample=0.82, colsample_bytree=0.5,
                reg_lambda=2.5, reg_alpha=0.15),
    "lgb": dict(num_leaves=255, min_child_samples=20, subsample=0.83, colsample_bytree=0.5, reg_lambda=3.0),
    "cat": dict(depth=8, l2_leaf_reg=5),
}

# What tuning chose in the notebook's run: only LightGBM's search beat its baseline.
# Used by `run_pipeline.py --skip-tuning`.
SELECTED_PARAMS = {
    "xgb": None,
    "lgb": dict(subsample=0.75, reg_lambda=6.0, num_leaves=383, min_child_samples=10, colsample_bytree=0.4),
    "cat": None,
}


# Hyperparameter tuning
N_ITER = 5   # random combinations tried per model

# Grids centred on the baseline values, one step down and one step up.
PARAM_GRIDS = {
    "xgb": {
        "max_leaves": [127, 255, 383],
        "min_child_weight": [10, 20, 40],
        "subsample": [0.75, 0.82, 0.9],
        "colsample_bytree": [0.4, 0.5, 0.6],
        "reg_lambda": [1.0, 2.5, 5.0],
        "reg_alpha": [0.05, 0.15, 0.5],
    },
    "lgb": {
        "num_leaves": [127, 255, 383],
        "min_child_samples": [10, 20, 40],
        "subsample": [0.75, 0.83, 0.9],
        "colsample_bytree": [0.4, 0.5, 0.6],
        "reg_lambda": [1.0, 3.0, 6.0],
    },
    "cat": {
        "depth": [7, 8, 9],
        "l2_leaf_reg": [3, 5, 8],
    },
}

# RandomizedSearchCV cannot do early stopping, so the search uses fixed tree counts
# close to where early stopping ended in the baseline run.
SEARCH_TREES = {"xgb": 5000, "lgb": 5300, "cat": 16000}

# Models trained on target-encoded features (CatBoost gets the ordinal codes only).
USES_TARGET_ENCODING = {"xgb": True, "lgb": True, "cat": False}
