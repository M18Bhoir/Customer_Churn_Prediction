# Customer Churn Prediction API

This repository implements a FastAPI service exposing an ML model for predicting customer churn. It includes a small model registry, prediction endpoint, health checks, and simple config-driven logging.

Key components

- `app/` — FastAPI application, routers, services, and models
- `model_registry/` — persisted trained model versions (expected layout: `model_YYYYMMDD_HHMMSS/model.pkl`)
- `data/` — raw and processed datasets used for training and experiments
- `requirements.txt` — Python dependencies

Overview

The API loads the latest model from `model_registry` at startup and exposes these endpoints under the API prefix configured in `app/core/config.yaml` (default `/api/v1`):

- `GET /api/v1/health` — basic health and model loaded status
- `GET /api/v1/model/version` — info about the loaded model
- `POST /api/v1/predict` — predict churn for a single customer (JSON body)

Quickstart — Local (development)

1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Ensure there is at least one model in `model_registry/` in a folder named like `model_YYYYMMDD_HHMMSS` and containing `model.pkl` (joblib or pickle). The app will look for the latest timestamped folder.

4. Run the API

```bash
uvicorn app.main:app --reload
```

By default the API listens on the host and port set in `app/core/config.yaml` (defaults: `localhost:8000`).

Configuration

The runtime configuration is located at `app/core/config.yaml`. Important keys:

- `application.APP_NAME`, `application.APP_VERSION`, `application.DEBUG`
- `api.API_PREFIX`, `api.HOST`, `api.PORT`
- `model.MODEL_REGISTRY_PATH` — path to the model registry directory
- `logging.LOG_LEVEL`, `logging.LOG_FILE`
- `prediction.PREDICTION_THRESHOLD`

Model registry

Place trained models in a timestamped folder under `model_registry/`. Example layout:

```
model_registry/
└── model_20260215_130847/
    ├── model.pkl
    ├── evaluation.json
    └── model.yaml
```

The server loads the latest folder by timestamp and attempts to load `model.pkl` using `joblib`.

API Endpoints

1) Health check

- Request: `GET {API_PREFIX}/health` (defaults to `/api/v1/health`)
- Response: JSON with `status`, `app_name`, `version`, `model_loaded`

Example:

```bash
curl -s http://localhost:8000/api/v1/health | jq
```

2) Model version

- Request: `GET {API_PREFIX}/model/version`
- Response: model status, version, type, loaded_at, model_file

3) Predict churn

- Request: `POST {API_PREFIX}/predict`
- Body: JSON matching the `CustomerData` schema. Required fields include `customer_id`, `age`, `tenure_months`, `monthly_charges`, `total_charges`, `contract_type`, `internet_service`, `support_calls`, `late_payments`.

Example request body:

```json
{
  "customer_id": 12345,
  "age": 35.0,
  "tenure_months": 24.0,
  "monthly_charges": 79.99,
  "total_charges": 1919.76,
  "contract_type": "One year",
  "internet_service": "Fiber optic",
  "support_calls": 2.0,
  "late_payments": 0
}
```

- Response: `PredictionResponse` with `customer_id`, `churn_prediction` (0/1), `churn_probability` (0-1), `risk_level`, and `model_version`.

Errors

- 400 — invalid input (validation errors)
- 503 — model not loaded
- 500 — server error during prediction

Logging

Logging is configured in `app/core/logger.py` and writes to the file configured in `app/core/config.yaml` (default `logs/app.log`). The logger also streams to stdout.

Docker

The repository includes a `Dockerfile` that builds a minimal image running `uvicorn`. Build and run:

```bash
docker build -t churn-api:latest .
docker run -p 8000:8000 --rm churn-api:latest
```

Notes for production

- Mount or copy a model into `model_registry/` inside the image or volume mount `model_registry/` at runtime.
- Tune `uvicorn` workers and process manager (e.g., Gunicorn + Uvicorn workers) for production.

Developer notes

- Schemas: `app/schemas/request.py` and `app/schemas/response.py`
- Prediction logic: `app/services/prediction_service.py`
- Model loading: `app/models/model_loader.py`

Tests

Basic tests are in `tests/`. Run with `pytest` after installing `pytest`.

Contact

For questions about this repo or model format, check the `notebooks/` and `src/` folders for training code and example notebooks.

---
