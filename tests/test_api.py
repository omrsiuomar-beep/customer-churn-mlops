import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"] in {"LogisticRegression", "RandomForest", "XGBoost"}
    assert body["model_version"] == "1.0.0"
    assert 0 < body["test_metrics"]["roc_auc"] <= 1


def test_predict(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk"] in {"low", "medium", "high"}
    assert body["churn"] == (body["churn_probability"] >= 0.5)


def test_risky_customer_scores_higher_than_loyal_one(client, customer):
    loyal = dict(customer, tenure=70, Contract="Two year", PaymentMethod="Credit card (automatic)",
                 InternetService="DSL", OnlineSecurity="Yes", TechSupport="Yes",
                 MonthlyCharges=55.0, TotalCharges=3850.0)
    risky = client.post("/predict", json=customer).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > safe


def test_missing_field(client, customer):
    customer.pop("Contract")
    assert client.post("/predict", json=customer).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("tenure", -3),
    ("MonthlyCharges", "cheap"),
    ("Contract", "Weekly"),
    ("SeniorCitizen", 2),
])
def test_invalid_values(client, customer, field, value):
    customer[field] = value
    assert client.post("/predict", json=customer).status_code == 422


def test_unknown_field(client, customer):
    customer["favourite_colour"] = "blue"
    assert client.post("/predict", json=customer).status_code == 422


def test_batch(client, customer):
    r = client.post("/predict-batch", json={"customers": [customer, customer, customer]})
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_empty_batch(client):
    assert client.post("/predict-batch", json={"customers": []}).status_code == 422
