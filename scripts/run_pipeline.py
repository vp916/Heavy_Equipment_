"""Runs the whole pipeline of notebooks/model_development.ipynb from the command line.

From the project root:

    python scripts/run_pipeline.py                  # full run, same flow as the notebook (about 2 hours)
    python scripts/run_pipeline.py --skip-tuning    # use the parameters tuning chose in the notebook (about 30-40 min)

Writes outputs/submission.csv and every model's out-of-fold and test predictions to outputs/.
"""
import argparse
import time

import numpy as np

import config
from features import prepare_data
from stack import stack_models, write_submission
from train import build_models, train_models
from tune import tune_all


def print_scores(title, val_scores):
    print(f"\n{title}")
    for name, scores in val_scores.items():
        print(f"{name.upper()} : {np.mean(scores):.5f}")


def main():
    parser = argparse.ArgumentParser(description="Heavy equipment price prediction pipeline")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="skip the baseline run and the search, and train once with the parameters "
                             "tuning chose in the notebook (config.SELECTED_PARAMS)")
    args = parser.parse_args()

    start = time.time()
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    print("Preparing the data...")
    X, y, y_log, X_test, test_ids, target_cols = prepare_data()
    print(f"X: {X.shape}, X_test: {X_test.shape}, target-encoded columns: {target_cols}")

    if args.skip_tuning:
        tuned = config.SELECTED_PARAMS
    else:
        print("\nStep 1: baseline training")
        _, _, _, val_scores = train_models(build_models(), X, y_log, X_test, target_cols)
        print_scores("Baseline Validation Scores", val_scores)
        baseline_scores = {name: np.mean(scores) for name, scores in val_scores.items()}

        print("\nStep 2: hyperparameter tuning")
        tuned = tune_all(X, y_log, baseline_scores, target_cols)

    print("\nStep 3: training with the chosen parameters")
    oof, test_preds, _, val_scores = train_models(build_models(tuned), X, y_log, X_test, target_cols)
    print_scores("Final Validation Scores", val_scores)

    for name in oof:
        np.save(config.OUTPUT_DIR / f"{name}_oof.npy", oof[name])
        np.save(config.OUTPUT_DIR / f"{name}_test.npy", test_preds[name])

    print("\nStep 4: stacking")
    stack_test, _ = stack_models(oof, test_preds, y_log)
    write_submission(test_ids, stack_test, config.OUTPUT_DIR / "submission.csv")

    print(f"\nDone in {(time.time() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
