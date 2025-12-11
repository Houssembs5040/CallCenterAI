"""
Transformer Service API
Simple FastAPI service for ticket classification
"""

import importlib.util
import os
import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# Add project root to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

spec = importlib.util.spec_from_file_location(
    "transformer_predictor", "models/transformer_predictor.py"
)
predictor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predictor_module)
TransformerPredictor = predictor_module.TransformerPredictor

# ============================================
# Prometheus Metrics
# ============================================
prediction_counter = Counter(
    "transformer_predictions_total", "Total number of predictions", ["status"]
)

prediction_latency = Histogram(
    "transformer_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

model_usage = Counter("transformer_model_usage", "Model usage counter", ["model_type"])

# ============================================
# Initialize FastAPI App
# ============================================
app = FastAPI(
    title="🤖 Transformer Classification Service",
    description="Advanced ticket classification using DistilBERT Multilingual",
    version="1.0.0",
)

# ============================================
# Load Model (happens once when service starts)
# ============================================
MODEL_PATH = "models/transformer"
LABEL_ENCODER_PATH = "models/transformer/label_encoder.pkl"

print("=" * 60)
print("🚀 Starting Transformer Service...")
print("=" * 60)

try:
    predictor = TransformerPredictor(MODEL_PATH, LABEL_ENCODER_PATH)
    print("✅ Transformer model loaded successfully!")
    print("=" * 60)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("=" * 60)
    predictor = None


# ============================================
# Define Request/Response Models
# ============================================
class PredictionRequest(BaseModel):
    """What the API expects from the user"""

    text: str
    return_probas: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "text": "My laptop screen is broken and needs repair",
                "return_probas": True,
            }
        }


class PredictionResponse(BaseModel):
    """What the API returns to the user"""

    text: str
    category: str
    confidence: float
    model: str
    all_probabilities: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "text": "My laptop screen is broken and needs repair",
                "category": "Hardware",
                "confidence": 0.98,
                "model": "distilbert-multilingual",
                "all_probabilities": {
                    "Hardware": 0.98,
                    "Storage": 0.01,
                    "Access": 0.01,
                },
            }
        }


# ============================================
# API Endpoints
# ============================================


@app.get("/")
def root():
    """
    Health check endpoint
    Visit: http://localhost:8000/
    """
    return {
        "service": "🤖 Transformer Classification Service",
        "status": "✅ healthy",
        "model": "DistilBERT Multilingual",
        "description": "Send POST request to /predict with ticket text",
    }


@app.get("/health")
def health():
    """
    Detailed health check
    Visit: http://localhost:8000/health
    """
    if predictor is None:
        return {"status": "❌ unhealthy", "error": "Model not loaded"}

    return {
        "status": "✅ healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "categories": list(predictor.label_encoder.classes_),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
        Predict ticket category

        Example request:
    ```json
        {
            "text": "My laptop is broken",
            "return_probas": true
        }
    ```

        Example response:
    ```json
        {
            "text": "My laptop is broken",
            "category": "Hardware",
            "confidence": 0.98,
            "model": "distilbert-multilingual"
        }
    ```
    """
    # Start timing
    start_time = time.time()

    # Check if model is loaded
    if predictor is None:
        prediction_counter.labels(status="error").inc()
        raise HTTPException(
            status_code=500, detail="Model not loaded. Please check server logs."
        )

    # Validate input
    if not request.text or len(request.text.strip()) == 0:
        prediction_counter.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Make prediction
        result = predictor.predict(request.text, return_probas=request.return_probas)

        # Add model info
        result["model"] = "distilbert-multilingual"

        # Record metrics
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        prediction_counter.labels(status="success").inc()
        model_usage.labels(model_type="transformer").inc()

        return result

    except Exception as e:
        prediction_counter.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/categories")
def get_categories():
    """
    Get list of available categories
    Visit: http://localhost:8000/categories
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    categories = list(predictor.label_encoder.classes_)

    return {"categories": categories, "total": len(categories)}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ============================================
# Run the server
# ============================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 Starting server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
