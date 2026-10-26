"""Ridge stacking of the three models' out-of-fold predictions, and the submission file."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import config
from train import rmsle


def stack_models(oof, test_preds, y_log):
    """Scores the Ridge stack with its own 5-fold CV, then fits the final Ridge on all out-of-fold predictions.

    Returns the stacked test predictions (log scale) and the final Ridge model."""
    names = ["xgb", "lgb", "cat"]
    train_stack = np.column_stack([oof[name] for name in names])
    test_stack = np.column_stack([test_preds[name] for name in names])

    stack_oof = np.zeros(len(y_log))
    stack_train_scores, stack_val_scores = [], []

    kf_stack = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    for fold, (train_idx, val_idx) in enumerate(kf_stack.split(train_stack), 1):

        model = Ridge(alpha=config.RIDGE_ALPHA)
        model.fit(train_stack[train_idx], y_log.iloc[train_idx])

        train_pred = model.predict(train_stack[train_idx])
        val_pred = model.predict(train_stack[val_idx])
        stack_oof[val_idx] = val_pred

        stack_train_scores.append(rmsle(y_log.iloc[train_idx], train_pred))
        stack_val_scores.append(rmsle(y_log.iloc[val_idx], val_pred))

        print(f"Fold {fold} | Train RMSLE: {stack_train_scores[-1]:.5f} | "
              f"Validation RMSLE: {stack_val_scores[-1]:.5f}")

    print("\nStacking Results")
    print(f"Average Train RMSLE      : {np.mean(stack_train_scores):.5f}")
    print(f"Average Validation RMSLE : {np.mean(stack_val_scores):.5f}")
    print(f"Overall OOF RMSLE        : {rmsle(y_log, stack_oof):.5f}")

    # final Ridge on all out-of-fold predictions
    final_meta_model = Ridge(alpha=config.RIDGE_ALPHA)
    final_meta_model.fit(train_stack, y_log)

    weights = pd.DataFrame({"Model": [name.upper() for name in names], "Ridge weight": final_meta_model.coef_})
    weights["Share (%)"] = 100 * weights["Ridge weight"] / weights["Ridge weight"].sum()
    print(f"\nIntercept: {final_meta_model.intercept_:.4f}")
    print(weights.round(3).to_string(index=False))

    return final_meta_model.predict(test_stack), final_meta_model


def write_submission(test_ids, stack_test, path):
    """Back to dollars with expm1, clipped at 0, one row per TransactionID."""
    final_predictions = np.maximum(np.expm1(stack_test), 0)

    submission = pd.DataFrame({"TransactionID": test_ids, "TargetValue": final_predictions})
    submission.to_csv(path, index=False)
    print(f"\nSaved {path} ({len(submission):,} rows)")
    return submission
