Here is a **clean, focused `README.md` specifically for running the `src/` application**.

You can place this at the **root of your repository**.

---

# ML Training Pipeline — Run Guide

This document explains how to run the complete ML pipeline located inside the `src/` directory.

The pipeline performs:

1. Data Loading
2. Data Validation
3. Feature Engineering
4. Processed Data Saving
5. Model Training
6. Evaluation
7. Model + Metadata Versioning

---

# Project Structure (Relevant to Running)

```
.
├── data/
│   ├── raw/
│   │   └── customer_churn.csv
│   └── processed/
│
├── src/
│   ├── config.yaml
│   ├── run.py
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   ├── validation_data.py
│   │   ├── build_features.py
│   │   └── transformers.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── schema/
│   ├── logger/
│   └── model_registry/
```

---

# Prerequisites

* Python 3.10+
* pip
* Virtual environment (recommended)

---

# 1️⃣ Setup Environment

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Mac / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a requirements file yet:

```bash
pip install pandas numpy scikit-learn pydantic pyyaml joblib
```

---

# 2️⃣ Configure the Pipeline

All configuration is controlled by:

```
src/config/config.yaml
```

Example:

```yaml
project:
  name: customer_churn_training

data:
  input_path: data/raw/customer_churn_dataset.csv
  processed_path: data/processed/processed_data.csv

  required_columns:
    - customer_id
    - age
    - tenure_months
    - monthly_charges
    - total_charges
    - contract_type
    - internet_service
    - support_calls
    - late_payments
    - churn

  target_column: churn

  allowed_missing_pct: 0.2


training:
  save_model: true
  model_registry_path: model_registry

  random_state: 42
  test_size: 0.2
  target_column: churn
  cross_validation_folds: 10

  model:
    type: logistic_regression   # options: logistic_regression | random_forest

    logistic_regression:
      penalty: l2
      C: 1.0
      solver: lbfgs
      max_iter: 1000
      class_weight: balanced
      n_jobs: -1

    random_forest:
      n_estimators: 200
      max_depth: 10
      min_samples_split: 5
      min_samples_leaf: 2
      max_features: sqrt
      bootstrap: true
      n_jobs: -1


evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc

```

Make sure your raw dataset exists at:

```
data/raw/customer_churn_dataset.csv
```

---

# 3️⃣ Run the Entire Pipeline

From the project root directory:

```bash
python -m src.run
```

---

# What Happens When You Run It

The following steps execute automatically:

### Step 1 — Load Raw Data

Reads from:

```
data/raw/
```

### Step 2 — Validate Data

Checks:

* Required columns
* Missing threshold
* Target integrity
* Duplicate rows

### Step 3 — Build Features

* Missing value imputation
* Feature generation
* Saves processed dataset to:

```
data/processed/processed_churn.csv
```

### Step 4 — Train Model

* Train/test split
* Model pipeline build
* Training

### Step 5 — Evaluate

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC

### Step 6 — Save Model Version

Creates:

```
src/model_registry/model_<timestamp>/
```

Inside it:

```
model.pkl
evaluation.json
model.yaml
predictions.csv
```

---

# Model Metadata (`model.yaml`)

Depending on model type:

### If RandomForest / Ensemble

Includes:

```yaml
feature_importance:
```

### If Logistic Regression

Includes:

```yaml
coefficients:
```

Also stores:

* Metrics
* Training timestamp
* Features used

---

# Processed Data Location

After feature engineering:

```
data/processed/processed_churn.csv
```

All training happens from this processed dataset.

---

