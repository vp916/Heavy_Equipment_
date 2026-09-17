# Heavy Equipment Selling Price Prediction

Machine Learning Project

## Table of Contents

- [Overview](#overview)
- [Challenge](#challenge)
- [Objective](#objective)
- [Evaluation Metric](#evaluation-metric)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Data Setup](#data-setup)

## Overview

This project focuses on predicting the selling price of heavy equipment using historical transaction data.

The project is based on the **Heavy Equipment Selling Price Prediction Challenge** as part of the Machine Learning Project.

## Challenge

The task is a supervised regression problem. Given information about heavy equipment and its transactions, the goal is to predict the selling price for unseen test observations.

The target variable is:

```text
TargetValue
```

## Objective

Build a machine learning solution that accurately predicts `TargetValue` for unseen heavy-equipment transactions.

## Evaluation Metric

The competition uses **Root Mean Squared Logarithmic Error (RMSLE)**.

$$
RMSLE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\log(1+\hat{y}_i) -
\log(1+y_i)
\right]^2
}
$$

where:

- $y_i$ is the actual selling price
- $\hat{y}_i$ is the predicted selling price
- $n$ is the number of observations

**Lower RMSLE indicates better performance.**

## Dataset

The competition provides training and test data along with supporting metadata.

Main files:

- `train.csv` — training data containing the target `TargetValue`
- `test.csv` — test data used for generating predictions
- `metadata.csv` — metadata describing the dataset and features
- `sample_submission.csv` — sample submission format, if provided

The competition datasets are not committed to this repository.

## Environment Setup

The project uses a dedicated Python virtual environment.

### Python Version

```text
Python 3.14
```

### Create the Environment

From the project root:

```bash
python -m venv .
```

This creates the virtual environment inside the project directory.

> **Note:** The virtual environment layout is different on Unix-based systems and Windows.

### Activate the Environment

#### Linux / macOS

For Bash / Zsh:

```bash
source ./bin/activate
```

For Fish shell:

```fish
source ./bin/activate.fish
```

#### Windows

For Command Prompt:

```cmd
.\Scripts\activate
```

For PowerShell:

```powershell
.\Scripts\Activate.ps1
```

### Upgrade pip

After activating the environment:

```bash
python -m pip install --upgrade pip
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Install Jupyter Kernel

Install `ipykernel`:

```bash
python -m pip install ipykernel
```

Register the environment as a Jupyter kernel.

The following command works in both Linux/macOS and Windows:

```bash
python -m ipykernel install --user --name heavy-equipment-project --display-name "Heavy Equipment Project (Python 3.14)"
```

The registered kernel can then be selected from VS Code or Jupyter when working with project notebooks.

### Verify the Environment

Run:

```python
import sys

print(sys.executable)
```

The output should point to the project's virtual environment.

For Linux/macOS:

```text
.../heavy_equipment/bin/python
```

For Windows:

```text
...\heavy_equipment\Scripts\python.exe
```

## Project Structure

The project is organized as follows:

```text
heavy_equipment/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebook/
│   └── model_development.ipynb
│
├── models/
│
└── test_result/
```

- `notebook/` — contains notebooks used for data analysis and model development.
- `models/` — stores trained models and related model artifacts.
- `test_result/` — stores prediction results and test outputs.
- `requirements.txt` — contains the Python dependencies required for the project.
- `.gitignore` — specifies files and directories that should not be tracked by Git.

The Python virtual environment is created inside the project directory, but its files are excluded from Git using `.gitignore`.

## Data Setup

The competition datasets are kept locally and are not committed to Git.

For local development, the downloaded competition files can be kept in the project directory:

```text
heavy_equipment/
├── train.csv
├── test.csv
├── metadata.csv
├── sample_submission.csv
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebook/
│   └── model_development.ipynb
│
├── models/
│
└── test_result/
```

The CSV files are ignored by Git through `.gitignore`.
