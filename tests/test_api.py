import json
import pytest
import requests


@pytest.fixture(scope="session")
def base_url():
    return "http://localhost:8000"


@pytest.fixture(scope="session")
def api_prefix():
    return "/api/v1"


def _request_safe(method, url, **kwargs):
    try:
        return requests.request(method, url, timeout=5, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        pytest.skip(f"Cannot connect to API at {url}: {exc}")


def _assert_prediction_structure(j):
    assert isinstance(j, dict), "Prediction response is not a JSON object"
    assert "risk_level" in j, "Missing 'risk_level' in prediction response"
    assert "churn_probability" in j, "Missing 'churn_probability' in prediction response"
    assert isinstance(j["churn_probability"], (float, int)), "'churn_probability' is not numeric"


def test_root(base_url):
    resp = _request_safe("GET", f"{base_url}/")
    assert resp.status_code == 200, f"Root endpoint returned {resp.status_code}: {resp.text}"
    assert isinstance(resp.json(), dict)


def test_health_check(base_url, api_prefix):
    resp = _request_safe("GET", f"{base_url}{api_prefix}/health")
    assert resp.status_code == 200, f"Health endpoint returned {resp.status_code}: {resp.text}"
    assert isinstance(resp.json(), dict)


def test_model_version(base_url, api_prefix):
    resp = _request_safe("GET", f"{base_url}{api_prefix}/model/version")
    assert resp.status_code == 200, f"Model version returned {resp.status_code}: {resp.text}"
    j = resp.json()
    assert isinstance(j, dict)
    assert any(k in j for k in ("version", "model_version", "model")), (
        "Model version response does not contain an expected version key"
    )


SAMPLE_CUSTOMER = {
    "customer_id": 12345,
    "age": 35.0,
    "tenure_months": 24.0,
    "monthly_charges": 79.99,
    "total_charges": 1919.76,
    "contract_type": "One year",
    "internet_service": "Fiber optic",
    "support_calls": 2.0,
    "late_payments": 0,
}


@pytest.mark.parametrize(
    "customer",
    [
        SAMPLE_CUSTOMER,
        {
            "customer_id": 1,
            "age": 25.0,
            "tenure_months": 3.0,
            "monthly_charges": 120.0,
            "total_charges": 360.0,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "support_calls": 5.0,
            "late_payments": 2,
        },
        {
            "customer_id": 2,
            "age": 55.0,
            "tenure_months": 60.0,
            "monthly_charges": 45.0,
            "total_charges": 2700.0,
            "contract_type": "Two year",
            "internet_service": "DSL",
            "support_calls": 0.0,
            "late_payments": 0,
        },
    ],
)
def test_predict_endpoint(base_url, api_prefix, customer):
    resp = _request_safe(
        "POST", f"{base_url}{api_prefix}/predict", json=customer
    )
    assert resp.status_code == 200, f"Predict returned {resp.status_code}: {resp.text}"
    j = resp.json()
    _assert_prediction_structure(j)
