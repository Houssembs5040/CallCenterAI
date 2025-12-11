import time

import joblib
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

model = joblib.load("models/tfidf/model_tfidf_svm.pkl")
vectorizer = joblib.load("models/tfidf/vectorizer_tfidf.pkl")

app = FastAPI(title="TF-IDF + SVM Ticket Classifier")

# ============================================
# Prometheus Metrics
# ============================================
REQUEST_COUNT = Counter(
    "tfidf_predictions_total", "Total number of predictions", ["status"]
)

PREDICTION_LATENCY = Histogram(
    "tfidf_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
)

MODEL_USAGE = Counter("tfidf_model_usage", "Model usage counter", ["model_type"])


class Ticket(BaseModel):
    text: str
    return_probas: bool = False


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "TF-IDF Classification Service",
        "status": "✅ healthy",
        "model": "TF-IDF + SVM",
    }


@app.get("/health")
def health():
    """Detailed health check"""
    return {"status": "✅ healthy", "model_loaded": True, "model": "TF-IDF + SVM"}


@app.post("/predict")
def predict(ticket: Ticket):
    """Predict ticket category"""
    start = time.time()

    try:
        # Transform and predict
        X = vectorizer.transform([ticket.text])
        pred = model.predict(X)[0]

        # Get probabilities if requested
        proba = None
        if ticket.return_probas and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0].max()

        latency = time.time() - start

        # Record metrics
        PREDICTION_LATENCY.observe(latency)
        REQUEST_COUNT.labels(status="success").inc()
        MODEL_USAGE.labels(model_type="tfidf").inc()

        result = {
            "text": ticket.text,
            "category": pred,
            "confidence": float(proba) if proba else 1.0,
            "model": "tfidf-svm",
            "latency": latency,
        }

        return result

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        return {"error": str(e), "status": "failed"}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
