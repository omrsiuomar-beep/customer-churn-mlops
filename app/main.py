import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import BatchRequest, BatchResponse, Customer, Prediction
from src.config import META_PATH, MODEL_PATH
from src.data import clean

state = {}


@asynccontextmanager
async def lifespan(app):
    # load once at startup, not on every request
    state["model"] = joblib.load(MODEL_PATH)
    state["meta"] = json.loads(META_PATH.read_text())
    yield
    state.clear()


app = FastAPI(
    title="Customer Churn API",
    description="Predicts whether a telecom customer is likely to cancel. Omar Yasser - IMT ML & MLOps Diploma.",
    version="1.0.0",
    lifespan=lifespan,
)


def risk_level(p):
    if p >= 0.6:
        return "high"
    if p >= 0.3:
        return "medium"
    return "low"


def predict_frame(customers):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    meta = state["meta"]
    frame = clean(pd.DataFrame([c.model_dump() for c in customers]))[meta["features"]]
    proba = state["model"].predict_proba(frame)[:, 1]
    threshold = meta.get("threshold", 0.5)
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            churn=bool(p >= threshold),
            risk=risk_level(p),
            model_version=meta["model_version"],
        )
        for p in proba
    ]


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading"}


@app.get("/model-info")
def model_info():
    meta = state.get("meta", {})
    keys = ["model_name", "model_version", "best_params", "threshold", "cv_f1", "cv_roc_auc", "test_metrics", "trained_at"]
    return {k: meta.get(k) for k in keys}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return predict_frame([customer])[0]


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    predictions = predict_frame(request.customers)
    return BatchResponse(predictions=predictions, count=len(predictions))
