"""Hyperparameter tuning with RandomizedSearchCV, leak-free."""
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

import config
from features import SmoothedTargetEncoder


def search_model(name):
    """The model as used during the search: no early stopping (the search has no validation set to watch),
    a fixed number of trees instead, and the baseline values of the tuned settings."""
    params = {k: v for k, v in config.FIXED_PARAMS[name].items() if k != "early_stopping_rounds"}
    trees = "iterations" if name == "cat" else "n_estimators"
    params[trees] = config.SEARCH_TREES[name]
    params.update(config.BASELINE_PARAMS[name])

    if name == "xgb":
        return xgb.XGBRegressor(**params)
    if name == "lgb":
        return lgb.LGBMRegressor(**params)
    return CatBoostRegressor(**params)


def tune_model(model_name, estimator, param_grid, X, y_log, kf, baseline_score,
               target_encode=False, target_cols=None, n_iter=3):
    """Returns the best parameters if their CV score beats the baseline, otherwise None."""
    print(f"\nTuning {model_name.upper()}...")

    if target_encode:
        # encoder + model as one estimator, so the encoder is refitted inside every CV fold
        estimator = Pipeline([
            ("encoder", SmoothedTargetEncoder(columns=target_cols, alpha=config.TARGET_ENCODING_ALPHA)),
            ("model", estimator),
        ])
        param_grid = {f"model__{k}": v for k, v in param_grid.items()}

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        random_state=config.SEED,
        n_jobs=1,      # every model already uses all CPU cores
        verbose=3,
    )
    search.fit(X.copy(), y_log)

    score = -search.best_score_
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    print(f"Baseline Score : {baseline_score:.5f}")
    print(f"Best CV Score  : {score:.5f}")
    print("Best Parameters:")
    print(best_params)

    if score < baseline_score:
        print("Using tuned parameters.\n")
        return best_params

    print("Keeping original parameters.\n")
    return None


def tune_all(X, y_log, baseline_scores, target_cols):
    """Tunes the three models. Returns {name: tuned parameters or None}."""
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    return {
        name: tune_model(name, search_model(name), config.PARAM_GRIDS[name], X, y_log, kf, baseline_scores[name],
                         target_encode=config.USES_TARGET_ENCODING[name], target_cols=target_cols,
                         n_iter=config.N_ITER)
        for name in ["xgb", "lgb", "cat"]
    }
