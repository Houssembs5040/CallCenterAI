import time

import joblib
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel

model = joblib.load("models/tfidf/model_tfidf_svm.pkl")
vectorizer = joblib.load("models/tfidf/vectorizer_tfidf.pkl")

app = FastAPI(title="TF-IDF + SVM Ticket Classifier")


REQUEST_COUNT = Counter("request_count", "Number of prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency per prediction")


class Ticket(BaseModel):
    text: str


@app.post("/predict")
def predict(ticket: Ticket):
    start = time.time()
    REQUEST_COUNT.inc()

    X = vectorizer.transform([ticket.text])
    pred = model.predict(X)[0]

    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)
    return {"category": pred, "latency": latency}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
