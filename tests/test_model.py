import json

import joblib
import pandas as pd
import pytest

from src.config import DATA_PATH, FEATURES, META_PATH, MODEL_PATH
from src.data import clean, load_data, split_data
from src.models import evaluate

MIN_ROC_AUC = 0.82
MIN_RECALL = 0.50

needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not downloaded")


def test_clean_fixes_total_charges():
    raw = pd.DataFrame([{**{c: "No" for c in FEATURES}, "customerID": "x", "SeniorCitizen": 1,
                         "tenure": 0, "MonthlyCharges": 20.0, "TotalCharges": " ", "Churn": "No"}])
    out = clean(raw)
    assert out.loc[0, "TotalCharges"] == 0.0
    assert out.loc[0, "SeniorCitizen"] == "Yes"
    assert "customerID" not in out.columns


def test_metadata_matches_features():
    meta = json.loads(META_PATH.read_text())
    assert meta["features"] == FEATURES
    assert meta["threshold"] == 0.5


@needs_data
def test_split_is_stratified():
    X_train, X_test, y_train, y_test = split_data(load_data())
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert X_train.isna().sum().sum() == 0


@needs_data
def test_quality_gate():
    model = joblib.load(MODEL_PATH)
    _, X_test, _, y_test = split_data(load_data())
    metrics = evaluate(y_test, model.predict_proba(X_test)[:, 1])
    assert metrics["roc_auc"] > MIN_ROC_AUC, f"ROC-AUC {metrics['roc_auc']:.3f} too low"
    assert metrics["recall"] > MIN_RECALL, f"recall {metrics['recall']:.3f} too low"
