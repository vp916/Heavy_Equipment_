# Heavy Equipment Selling Price Prediction

**Gradient-boosted ensembles with leak-aware target encoding for pricing used industrial machinery from sparse, high-cardinality tabular data**

## Abstract

This project predicts the sale price (`TargetValue`) of used heavy equipment such as backhoe loaders, motor graders, track excavators, track-type tractors and wheel loaders. Each record has 50 transactional, geographic and technical attributes.

The data is unusually hostile to modelling:

- **47.9% of all cells are missing, and not one of the 138,701 training rows is complete.** Another 534,061 cells hold the disguised placeholder `"None or Unspecified"`.
- **The model code has 3,505 levels, and half of them occur five times or fewer.**
- **Most columns are anonymised** (`col1`–`col30`), and **10.6% of manufacture years are a placeholder**.
- **The data has a hard noise floor.** Even machines with the same model code and the same manufacture year differ in price by about 0.30 in log space. Rows with identical features are sometimes sold at different prices.

Section 2 quantifies each of these problems.

The pipeline:

1. Explores and visualises the data.
2. Builds features from dates, machine age, usage hours, the functional classification label and measurement strings.
3. Trains XGBoost, LightGBM and CatBoost with 5-fold cross-validation. Smoothed target encoding is fitted inside each training fold.
4. Tunes each model with `RandomizedSearchCV`.
5. Combines the three models' out-of-fold predictions with a Ridge meta-model (stacking).
6. Writes `submission.csv` for the test set.

The work was split by model. Each model was first developed and tuned in its own notebook (kept in [`notebooks/versions/`](notebooks/versions/)). The three were then combined in [`notebooks/model_development.ipynb`](notebooks/model_development.ipynb).

| | |
|---|---|
| **Task** | Tabular regression · 138,701 training and 15,000 test transactions |
| **Metric** | Root Mean Squared Logarithmic Error (lower is better) |
| **Models** | XGBoost, LightGBM, CatBoost → Ridge stacking (5-fold) |
| **Best single model (5-fold CV)** | LightGBM (tuned): 0.20169 |
| **Stacked ensemble (5-fold OOF)** | **0.19805** |
| **Leaderboard (held-out test)** | **0.19296** |

![Cross-validation RMSLE across milestones](assets/experiment_progression.png)

The blue line shows the leak-free 5-fold CV score at each milestone. The hollow markers show what the hyperparameter search reported. The search fits the target encoder on all rows, so validation rows have already contributed to their own encodings. Once the models were retrained with the encoder fitted inside each fold, most of the apparent tuning gain disappeared. The largest real improvement came from stacking the three models.

---

## Table of Contents

1. [Problem](#1-problem)
2. [Dataset](#2-dataset)
3. [Evaluation Metric](#3-evaluation-metric)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Observations and Limitations](#6-observations-and-limitations)
7. [Project Structure](#7-project-structure)
8. [Environment Setup](#8-environment-setup)
9. [Data Setup](#9-data-setup)
10. [Running the Notebook](#10-running-the-notebook)

---

## 1. Problem

Each row is one completed sale of a piece of heavy machinery. A row contains system metadata, the sale region, the machine's manufacture year and hour-meter reading, a hierarchy of specification codes, and up to 30 component attributes such as cab type, drivetrain, tyre size, track width and arm reach.

The goal is to predict `TargetValue`, the sale amount in US dollars, for every `TransactionID` in the test set. This is a supervised regression problem.

---

## 2. Dataset

| File | Rows | Columns | Description |
|---|---:|---:|---|
| `train.csv` | 138,701 | 50 | Transactions including the target `TargetValue` |
| `test.csv` | 15,000 | 49 | Transactions to predict |
| `metadata.csv` | 50 | 2 | Plain-language description of every column |

### Feature groups

| Group | Columns | Notes |
|---|---|---|
| Identifiers | `TransactionID`, `AssetID`, `ProductConfigID` | `AssetID` has 120,111 values (nearly unique); `ProductConfigID` has 3,594 |
| Source and partner | `DataOriginCode`, `VendorPartnerID` | 6 and 31 values |
| Time | `TransactionDate`, `ManufactureYear` | Sales from 1990 to 2013 |
| Usage | `OperationalHoursMeter`, `UtilizationTier` | 40.8% and 63.6% missing |
| Specification | `Spec_FullDescriptor`, `Spec_BaseClass`, `Spec_SubClass`, `Spec_ReleaseSeries`, `Spec_VariantModifier` | 3,505 / 1,249 / 147 / 113 / 132 values |
| Classification | `FunctionalClassification`, `InventoryGroupCategory`, `InventoryGroupDescription`, `AssetScaleFactor` | e.g. *"Wheel Loader - 150.0 to 175.0 Horsepower"* |
| Location | `RegionCode` | 53 regions |
| Components | `CabinType`, `DrivetrainType`, `Forks`, `col1`–`col30` | Anonymised columns; `metadata.csv` describes each one |

### Difficulty at a glance

| Property | Measurement (training set) |
|---|---|
| Missing cells | **47.9%** of all cells (48.8% in test); **0** complete rows |
| Columns more than 50% / 80% empty | **24 / 15** of 50 |
| Disguised missing values | **534,061** cells equal to `"None or Unspecified"`, across 20 columns |
| Usable hour-meter readings | **36.4%** of rows (40.8% missing, 22.8% exactly 0) |
| Placeholder manufacture year | **14,719** rows (10.6%) with year 1001 |
| Model-code levels (`Spec_FullDescriptor`) | **3,505**: 776 seen once, 50% seen five times or fewer |
| Configuration levels (`ProductConfigID`) | **3,594**: 806 seen once, 51% seen five times or fewer |
| Model codes in test never seen in training | **80** |
| Anonymised component columns | **27** (`col1`–`col30`, with `col2`, `col17` and `col26` absent) |
| Target precision | **675** distinct prices; 97.9% are multiples of \$500 |
| Spread of log price within one model code + manufacture year | **0.30** (compared with 0.67 for the whole dataset) |

### Data challenges

**1. Skewed target.** Prices range from \$7,500 to \$142,000, with a median of \$35,000 and a mean of \$41,522. The distribution has a long right tail. `TargetValue` has only 675 distinct values, so sale prices appear to be rounded. Taking the log of the target makes the distribution much closer to normal.

![Target distribution](assets/target_distribution.png)

**2. Placeholder years.** 14,719 rows (10.6%) have `ManufactureYear = 1001`. This value means *unknown*. It is not a real year. If it were used as-is, those machines would appear to be about 1,000 years old.

![Manufacture year distribution](assets/manufacture_year_distribution.png)

**3. Structured missingness.** Component columns are not missing at random. Each block of columns is filled in for one inventory group only:

| Inventory group | Rows | Component columns specific to this group |
|---|---:|---|
| Track Excavators (TEX) | 71,018 | `col20`–`col25` |
| Motor Graders (MG) | 23,405 | `col6`–`col9`, `col11`, `col13`, `col14` |
| Wheel Loader (WL) | 18,872 | `col29`, `col30` |
| Track Type Tractors (TTT) | 14,959 | `col27`, `col28` |
| Backhoe Loaders (BL) | 10,381 | `col4`, `col5` |
| Skid Steer Loaders (SSL) | 66 | `col18`, `col19` |

![Missing data map](assets/missing_data_map.png)

A missing component value therefore usually means *not applicable to this machine type*. The pipeline keeps this as its own category instead of dropping the columns.

**4. A long tail of rare categories.** The specification codes follow a long-tailed distribution. Half of the 3,505 model codes appear five times or fewer, and 776 appear only once. There is too little data to learn a reliable price for most of them. Simple ordinal codes carry no notion of price, and naive per-category averages overfit badly. This is why the pipeline uses *smoothed* target encoding fitted inside each fold. In addition, 80 model codes in the test set never appear in training, so the model must fall back to broader groupings for them.

**5. Missing values in disguise.** Real `NaN`s are only part of the problem. 534,061 cells across 20 columns contain the text `"None or Unspecified"`, a second kind of missing value that looks like a real category. The hour meter has the same issue. 40.8% of readings are missing and another 22.8% are exactly 0, so only about a third of machines have a usable usage figure. The surviving readings are themselves unreliable: the 99th percentile is 25,796 hours, but the 99.9th is 412,138 and the maximum is 1,729,600 hours, which is about 197 years of non-stop running.

**6. Anonymised and loosely documented features.** 27 component columns are named only `col1`–`col30`, and three numbers in that range are missing from the schema. `metadata.csv` gives only short, paraphrased descriptions such as *"The type of interface used for load-bearing and shock absorption"*, with no units or allowed values. Measurements mix formats and units inside a single column: `<12'`, `13"`, `26.5`, `16 inch`, `10' 2"`. Labels are inconsistent (`AutoShift` vs `Autoshift`). The data also contains impossible records: 7 machines are sold before their manufacture year, and 21 have manufacture years between 1920 and 1949.

**7. A hard noise floor.** Much of the price variation cannot be explained by the available features:

| Machines grouped by | Spread of log price within a group |
|---|---:|
| Nothing (whole dataset) | 0.671 |
| Model code | 0.327 |
| Model code + manufacture year | 0.300 |
| Model code + manufacture year + sale year + region | 0.144 |

Even when model code, manufacture year, sale year and region all match, the price still varies by about 0.14 in log space. That is roughly ±15% for otherwise identical machines. Only 46% of rows have such a match at all. 6,792 rows are exact duplicates of another row on every feature. In 71% of these 2,954 duplicate groups the machines sold at different prices, in one case 4.7 times apart. No model can resolve that difference. For comparison, the final ensemble reaches an RMSLE of 0.198.

**8. A changing market.** The median sale price moves with the economic cycle: \$31,000 in 1990, \$43,000 in 1992, \$32,500 in 2003, \$40,000 in 2007 and \$31,500 in 2010. The same machine is worth different amounts in different years. Sale-date features are therefore essential, and the model has to separate the market trend from depreciation.

---

## 3. Evaluation Metric

The project uses **Root Mean Squared Logarithmic Error (RMSLE)**.

```math
\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(\log(1+\hat{y}_i) - \log(1+y_i)\big)^2}
```

where $y_i$ is the actual selling price, $\hat{y}_i$ is the predicted selling price and $n$ is the number of observations. **Lower RMSLE is better.**

RMSLE measures **relative** error. A \$5,000 miss on a \$20,000 machine counts for more than the same miss on a \$120,000 machine. RMSLE equals the RMSE of $\log(1+y)$, so every model is trained with a squared-error objective on `log1p(TargetValue)`. Predictions are converted back with `expm1` and clipped at 0.

---

## 4. Methodology

```text
train.csv / test.csv
      │
      ▼
parse_functional_classification ─► machine family, drive type, capacity range, unit
clean_machine_measurements      ─► col7 / col15 / col21 / col22 as numbers
feature_engineering             ─► date parts, asset age, missing flags, "Not Applicable" fills
DataTransformer                 ─► median imputation, log hours, hours per year, ordinal encoding
      │
      ▼
5-fold KFold ── for each fold ───────────────────────────────────────┐
  SmoothedTargetEncoder (fitted on the training fold only)            │
  XGBoost  (ordinal + target-encoded features)                        │
  LightGBM (ordinal + target-encoded features)                        │
  CatBoost (ordinal features)                                         │
      │                                                               │
      ▼                                                               │
Out-of-fold predictions ─► Ridge meta-model (5-fold) ─► submission.csv
```

### 4.1 Feature engineering

| Feature | How it is built | Why |
|---|---|---|
| `TransactionYear`, `Month`, `Quarter`, `Day`, `DayOfWeek` | From `TransactionDate` | Captures market level and seasonality |
| `ManufactureYearMissing` | Year 1000 or 1001 → flag = 1 and year → `NaN` | Separates *unknown* from *old* |
| `AssetAge` | `TransactionYear − ManufactureYear`; negative values → `NaN` | Depreciation drives price |
| `HoursMissing` | Flag for a missing hour-meter reading | Missing hours can carry information |
| `OperationalHoursLog` | `log1p(hours)` after median imputation | Reduces the effect of extreme readings |
| `HoursPerYear` | `hours / (AssetAge + 1)` | Usage intensity, independent of age |
| `FC_MachineFamily`, `FC_DriveType` | Parsed from `FunctionalClassification` | Shared structure across model codes |
| `FC_LowerBound`, `FC_UpperBound`, `FC_Unit`, `FC_IsPlusRange`, `FC_RangeKnown` | Parsed capacity range, e.g. *150.0 to 175.0 Horsepower* | Turns a text label into an ordered size value |
| `col7_Numeric_Feet`, `col15_Numeric_Inches`, `col21_Numeric_Inches`, `col22_Total_Inches` | Parsed measurement strings | Lets tree models use physical size |
| `Missing_BaseClass` | Flag for a missing `Spec_BaseClass` | Missingness indicator |
| Component columns (`col*`) | Missing values filled with `"Not Applicable"` | Keeps structured missingness as a category |

**Dropped columns:** `TransactionID`, `AssetID` (nearly unique), `InventoryGroupCategory` (duplicates `InventoryGroupDescription`), and `col18` / `col19` (99.95% missing).

### 4.2 Encoding

- **`DataTransformer`** fills missing numeric values with the median. It then ordinal-encodes every categorical column. Unseen categories become −1.
- **`SmoothedTargetEncoder`** is applied to the 8 categorical columns with more than 20 levels: `VendorPartnerID`, `Spec_FullDescriptor`, `Spec_BaseClass`, `Spec_SubClass`, `Spec_ReleaseSeries`, `Spec_VariantModifier`, `FunctionalClassification` and `RegionCode`. It is computed on the log target:

```math
\text{TE}(c) = \frac{n_c\,\bar{y}_c + \alpha\,\bar{y}}{n_c + \alpha}, \qquad \alpha = 15
```

Rare categories are pulled toward the global mean $\bar{y}$. Unseen categories get $\bar{y}$. During training, the encoder is **fitted on each training fold only** and then applied to that fold's validation rows and to the test set. XGBoost and LightGBM use the target-encoded features. CatBoost uses the ordinal-encoded features only.

### 4.3 Models

All three models use up to 12,000 trees, a learning rate of 0.018, seed 42 and early stopping after 300 rounds without improvement on the validation fold.

| Model | Baseline parameters | Search grid (`RandomizedSearchCV`, 5 candidates × 5 folds) |
|---|---|---|
| **XGBoost** | depth 8, min child weight 5, subsample 0.85, colsample 0.85, λ 3, α 0.1, `hist` | depth {7, 8, 9}, min child weight {4, 5, 6}, lr {0.018, 0.020, 0.022}, subsample / colsample {0.82, 0.85, 0.88}, α {0.05, 0.10, 0.15}, λ {2.5, 3.0, 4.0} |
| **LightGBM** | 80 leaves, min child samples 30, subsample 0.85, colsample 0.85, λ 3 | leaves {72, 80, 88}, min child samples {25, 30, 35}, lr {0.018, 0.020, 0.022}, subsample / colsample {0.83, 0.85, 0.87}, λ {2.5, 3.0, 3.5} |
| **CatBoost** | depth 8, L2 5, RMSE loss | depth {7, 8, 9}, lr {0.018, 0.020, 0.022}, L2 {4, 5, 6} |

A model's tuned parameters are used only if the best search score beats its baseline CV score.

### 4.4 Validation and stacking

- **5-fold shuffled `KFold`** (seed 42). This gives an out-of-fold (OOF) prediction for every training row. Test predictions are averaged over the 5 folds.
- **Ridge regression** (α = 1.0) is the meta-model. It is trained on the three models' OOF log-predictions and scored with its own 5-fold CV. It is then refitted on all OOF predictions to produce the final test predictions.

---

## 5. Results

### 5.1 Baseline models (5-fold CV RMSLE)

| Fold | XGBoost | LightGBM | CatBoost |
|---:|---:|---:|---:|
| 1 | 0.19934 | 0.19988 | 0.20453 |
| 2 | 0.20119 | 0.20206 | 0.20700 |
| 3 | 0.20337 | 0.20354 | 0.20830 |
| 4 | 0.20264 | 0.20298 | 0.20894 |
| 5 | 0.20150 | 0.20230 | 0.20617 |
| **Mean** | **0.20161** | **0.20215** | **0.20699** |
| Std | 0.00138 | 0.00125 | 0.00156 |

![Validation RMSLE across folds](assets/validation_rmsle_across_folds.png)

### 5.2 Hyperparameter tuning

| Model | Baseline CV | Best search score | Decision | Best parameters |
|---|---:|---:|---|---|
| XGBoost | 0.20161 | 0.19675 | Tuned parameters used | depth 9, lr 0.018, min child weight 4, subsample 0.82, colsample 0.88, λ 2.5, α 0.15 |
| LightGBM | 0.20215 | 0.19800 | Tuned parameters used | 88 leaves, lr 0.02, min child samples 25, subsample 0.83, colsample 0.83, λ 3.0 |
| CatBoost | 0.20699 | 0.20769 | Baseline kept | (depth 9, lr 0.018, L2 5 was the best candidate) |

### 5.3 Retraining with the selected parameters and stacking

| Model | 5-fold CV RMSLE |
|---|---:|
| XGBoost (tuned) | 0.20189 |
| LightGBM (tuned) | 0.20169 |
| CatBoost (baseline) | 0.20699 |
| **Ridge stack** | **0.19805** (OOF) · 0.19804 (mean of folds) |
| **Ridge stack, leaderboard** | **0.19296** (held-out test) |

Stacking improves on the best single model (LightGBM, 0.20169) by about 0.0036 RMSLE.

On the leaderboard the stacked submission scores **0.19296**, about 0.005 better than its CV score. Cross-validation is therefore a conservative estimate. If target information were leaking into the validation folds, the CV score would be the optimistic one instead.

---

## 6. Observations and Limitations

- **The tuning scores are optimistic.** Inside the tuning function, the target encoder is fitted on the full training set before `RandomizedSearchCV` runs. Each validation fold has therefore already contributed to its own encodings. After retraining with in-fold encoding, the large search gains mostly disappear. XGBoost goes from 0.20161 to 0.20189, which is slightly worse. LightGBM goes from 0.20215 to 0.20169. Fitting the encoder inside the search (for example in a scikit-learn `Pipeline`) would give more reliable tuning scores.
- **Minor preprocessing leakage.** The median imputer and the ordinal encoder are fitted on the full training set before cross-validation. The effect is likely small.
- **LightGBM did not converge.** LightGBM used nearly all 12,000 trees in every fold. A higher learning rate or a larger tree budget may help.
- **CatBoost is the weakest model.** It receives ordinal codes as plain numbers (no `cat_features`). Its native categorical handling was not used in this pipeline.
- **Small tuning search.** Only 5 random candidates per model were tried, from narrow grids around the baseline.
- **Shuffled K-fold mixes years.** Validating on later sale dates would better match how a pricing model is used.
- **Only the final ensemble has a leaderboard score.** The single-model and tuning results are cross-validation scores only.
- **Pandas 3 skips the `"Unknown"` fill.** Selecting columns with `dtypes == "object"` finds no columns under pandas 3, so missing descriptive values are encoded as −1 instead. Using `select_dtypes(include=["object", "string"])` would restore the intended behaviour.

---

## 7. Project Structure

```text
heavy_equipment/
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/                                  # figures used in this README
│   ├── experiment_progression.png
│   ├── target_distribution.png
│   ├── manufacture_year_distribution.png
│   ├── missing_data_map.png
│   └── validation_rmsle_across_folds.png
│
├── notebooks/
│   ├── model_development.ipynb              # final pipeline: EDA → features → 3 models → tuning → stacking → submission
│   └── versions/                            # milestone notebooks (one model each)
│       ├── README.md
│       ├── training_xgboost.ipynb
│       ├── training_lightgbm.ipynb
│       └── training_catboost.ipynb
│
└── models/                                  # saved predictions / model artifacts (not tracked)
```

- `notebooks/model_development.ipynb` is the main notebook. It combines the three models, their tuning and the stacking step, and writes `submission.csv`.
- `notebooks/versions/` holds the single-model notebooks the final pipeline was built from. They share the same EDA and feature-engineering steps, and each one trains and tunes one model. The team developed them in parallel, one notebook per member. See [`notebooks/versions/README.md`](notebooks/versions/README.md) for their scores.

The datasets (`*.csv`), model files, CatBoost logs (`catboost_info/`) and the virtual environment are excluded from Git through `.gitignore`.

---

## 8. Environment Setup

The project uses a Python virtual environment created inside the project directory.

### Python version

```text
Python 3.14
```

### Create the environment

From the project root:

```bash
python -m venv .
```

> **Note:** The virtual environment layout differs between Unix-based systems (`bin/`) and Windows (`Scripts/`).

### Activate the environment

| Platform / shell | Command |
|---|---|
| Linux / macOS (bash, zsh) | `source ./bin/activate` |
| Linux / macOS (fish) | `source ./bin/activate.fish` |
| Windows (Command Prompt) | `.\Scripts\activate` |
| Windows (PowerShell) | `.\Scripts\Activate.ps1` |

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main libraries are `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `matplotlib` and `seaborn`. `requirements.txt` pins exact versions.

### Register the Jupyter kernel

```bash
python -m ipykernel install --user --name heavy-equipment-project --display-name "Heavy Equipment Project (Python 3.14)"
```

Select this kernel in VS Code or Jupyter when you work with the project notebooks.

### Verify the environment

```python
import sys
print(sys.executable)
```

The output should point to the project's virtual environment:

- Linux / macOS: `.../heavy_equipment/bin/python`
- Windows: `...\heavy_equipment\Scripts\python.exe`

---

## 9. Data Setup

The competition datasets are kept locally and are not committed to Git. Place `train.csv`, `test.csv` and `metadata.csv` in the **project root**. The notebooks read them from there (`../train.csv` relative to `notebooks/`).

```text
heavy_equipment/
├── train.csv
├── test.csv
├── metadata.csv
├── README.md
├── requirements.txt
└── notebooks/ ...
```

---

## 10. Running the Notebook

With the virtual environment activated, open `notebooks/model_development.ipynb` in VS Code or Jupyter, select the project kernel and run all cells.

The notebook:

1. Loads and explores the data.
2. Builds features and encodes the data.
3. Trains the baseline XGBoost, LightGBM and CatBoost models with 5-fold CV.
4. Tunes each model with `RandomizedSearchCV`.
5. Retrains with the selected parameters.
6. Fits the Ridge stacking model.
7. Writes `notebooks/submission.csv` (15,000 rows: `TransactionID, TargetValue`).

Training runs on the CPU. No GPU is needed. Hyperparameter tuning is the slowest step: each fit takes between about 30 seconds and 1.7 minutes, and each model runs 25 fits.

### Output of a full run

**Stacking.** The Ridge meta-model scores the same on its training and validation folds (both average 0.19804). This means the stacking step itself does not overfit:

| Fold | Train RMSLE | Validation RMSLE |
|---:|---:|---:|
| 1 | 0.19866 | 0.19556 |
| 2 | 0.19815 | 0.19761 |
| 3 | 0.19759 | 0.19986 |
| 4 | 0.19770 | 0.19939 |
| 5 | 0.19810 | 0.19780 |
| **Overall OOF** | | **0.19805** |

**Early stopping.** In the retraining run, the tuned XGBoost stopped after 5,566 to 6,416 trees. LightGBM used 11,860 to 11,998 of its 12,000 trees in every fold, so it was still improving when the budget ran out.

**Submission.** `submission.csv` has 15,000 rows with one prediction per `TransactionID` and no missing values. The predicted prices follow the shape of the training target closely:

| | Min | 25% | Median | Mean | 75% | Max |
|---|---:|---:|---:|---:|---:|---:|
| Training `TargetValue` | \$7,500 | \$20,500 | \$35,000 | \$41,522 | \$57,000 | \$142,000 |
| Predicted (test) | \$7,226 | \$20,873 | \$34,578 | \$40,923 | \$56,115 | \$144,110 |

The predicted distribution is slightly narrower (standard deviation \$25,050 vs \$26,361). This is expected: averaging three models and five folds pulls the most extreme predictions toward the middle.
