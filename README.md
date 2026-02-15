Below is a **production-quality `README.md`** suitable for a **data science + FastAPI monolith ML project**.
It explains architecture, setup, execution flow, and how to run the full training pipeline.

You can place this at the **root of your repository**.

---

# Customer Churn Prediction — Data Science Monolith (FastAPI Ready)

## Overview

This repository contains an **industrial-standard data science project structure** designed for:

* End-to-end ML training pipelines
* Feature engineering workflows
* Model versioning
* Evaluation tracking
* Production deployment readiness using FastAPI

The project follows a **monolith architecture**, where:

* Data ingestion
* Validation
* Feature engineering
* Model training
* Evaluation
* Model registry management

are all managed inside a single structured application.

---

## Project Architecture

```
.
├── data/
│   ├── raw/
│   │   └── customer_churn_dataset.csv
│   └── processed/
│       └── processed_churn.csv
│
├── notebooks/
│   ├── feature_engineering/
│   ├── feature_selection/
│   ├── modeling/
│   └── evaluation/
│
├── src/
│   ├── config.yaml
│   ├── run.py
│   │
│   ├── logger/
│   │   └── logger.py
│   │
│   ├── schema/
│   │   ├── data_schema.py
│   │   ├── train_schema.py
│   │   ├── predict_schema.py
│   │   └── evaluate_schema.py
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
│   └── model_registry/
│       └── model_<timestamp>/
│           ├── model.pkl
│           ├── evaluation.json
│           └── model.yaml
│
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Pipeline Flow

The pipeline executes in the following order:

```
Raw Data
   ↓
Data Loading
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Processed Data Saved
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation
   ↓
Model + Metadata Saved
```

---

## Model Registry

Each training run creates a new version:

```
model_registry/
└── model_YYYYMMDD_HHMMSS/
    ├── model.pkl
    ├── evaluation.json
    └── model.yaml
```

### model.yaml contains

* Model type
* Training timestamp
* Features used
* Feature importance (ensemble models)
* Coefficients (linear models)
* Evaluation metrics

---

## Requirements

* Python 3.10+
* pip or conda
* Docker (optional)

---

## Installation

### 1️⃣ Clone Repository

```bash
git clone <repo-url>
cd churn-prediction
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

All runtime configuration is controlled via:

```
src/config.yaml
```

Key configurations:

```yaml
data:
  input_path: data/raw/customer_churn_synthetic_100k.csv
  processed_path: data/processed/processed_churn.csv

training:
  model_type: random_forest
  target_column: churn
  test_size: 0.2
```

---

## Running the Training Pipeline

Run the entire pipeline using:

```bash
python src/run.py
```

This will:

1. Load raw dataset
2. Validate data schema
3. Build features
4. Save processed dataset
5. Train model
6. Evaluate model
7. Save model artifacts in model_registry

---

## Running with Docker

### Build Image

```bash
docker build -t churn-model .
```

### Run Container

```bash
docker run churn-model
```

or

```bash
docker-compose up --build
```

---

## Feature Engineering

Feature engineering happens inside:

```
src/data/build_features.py
```

Custom transformers are defined in:

```
src/data/transformers.py
```

Processed data is stored in:

```
data/processed/
```

All model training uses processed data only.

---

## Notebooks

The `notebooks/` directory is reserved for:

* Feature exploration
* Feature selection
* Modeling experiments
* Evaluation analysis

Notebook code should eventually be migrated into `src/`.

---

## Logging

Centralized logging is implemented in:

```
src/logger/logger.py
```

All modules use the same logging interface.
