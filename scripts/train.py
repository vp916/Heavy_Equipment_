"""The three models and the 5-fold cross-validated training loop."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

import config
from features import SmoothedTargetEncoder


def build_model(name, tuned_params=None):
    """One model with its fixed settings plus either the tuned or the baseline values."""
    params = {**config.FIXED_PARAMS[name], **(tuned_params or config.BASELINE_PARAMS[name])}

    if name == "xgb":
        return xgb.XGBRegressor(**params)
    if name == "lgb":
        return lgb.LGBMRegressor(**params)
    return CatBoostRegressor(**params)


def build_models(tuned=None):
    """All three models. `tuned` maps a model name to its tuned parameters (None = keep the baseline)."""
    tuned = tuned or {}
    return {name: build_model(name, tuned.get(name)) for name in ["xgb", "lgb", "cat"]}


def rmsle(y_log_true, y_log_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_log_true), np.expm1(y_log_pred)))


def train_models(models, X, y_log, X_test, target_cols):
    """5-fold CV. In every fold the target encoder is fitted on the training part only, each model trains with
    early stopping on the validation part, and the validation predictions are kept as out-of-fold predictions.
    The test predictions are averaged over the 5 fold models."""
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    oof = {name: np.zeros(len(X)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}
    train_scores = {name: [] for name in models}
    val_scores = {name: [] for name in models}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):

        print(f"\nFold {fold}")

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y_log.iloc[train_idx]
        y_val = y_log.iloc[val_idx]

        # target encoding, fitted on this fold's training part only (used by XGBoost and LightGBM)
        encoder = SmoothedTargetEncoder(columns=target_cols, alpha=config.TARGET_ENCODING_ALPHA)
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_val_enc = encoder.transform(X_val)
        X_test_enc = encoder.transform(X_test)

        for name, model in models.items():

            if name == "cat":
                model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                test_preds[name] += model.predict(X_test) / kf.n_splits

            elif name == "lgb":
                model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)],
                          callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)])

                train_pred = model.predict(X_train_enc)
                val_pred = model.predict(X_val_enc)
                test_preds[name] += model.predict(X_test_enc) / kf.n_splits
                print(f"LGB trees kept: {model.n_estimators_}")

            else:   # XGBoost
                model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=500)

                train_pred = model.predict(X_train_enc)
                val_pred = model.predict(X_val_enc)
                test_preds[name] += model.predict(X_test_enc) / kf.n_splits

            oof[name][val_idx] = val_pred

            train_scores[name].append(rmsle(y_train, train_pred))
            val_scores[name].append(rmsle(y_val, val_pred))

            print(f"{name.upper()} : {val_scores[name][-1]:.5f}")

    return oof, test_preds, train_scores, val_scores
