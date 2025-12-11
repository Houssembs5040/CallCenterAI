import asyncio
import json
import logging
import os
import re
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

# --------------------------------------------
# Prometheus Metrics
# --------------------------------------------
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# Total prediction requests handled by the agent
AGENT_REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total number of prediction requests processed by the agent",
    ["status"],
)

# Routing decisions (TF-IDF vs Transformer)
AGENT_ROUTING_DECISIONS = Counter(
    "agent_routing_decisions_total",
    "Total routing decisions made by the agent",
    ["model"],
)

# Latency for agent decision + upstream call
AGENT_LATENCY = Histogram(
    "agent_prediction_latency_seconds",
    "End-to-end prediction latency for the agent in seconds",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
)

# PII detection counters
AGENT_PII_DETECTED = Counter(
    "agent_pii_detected_total",
    "Total PII occurrences detected and scrubbed by the agent",
    ["pii_type"],
)

# ============================================
# Initialize FastAPI App
# ============================================
app = FastAPI(
    title="🤖 AI Agent Service",
    description="Intelligent routing to the best classification model",
    version="1.0.0",
)

# ============================================
# Service Configuration
# ============================================
TRANSFORMER_SERVICE = "http://transformer-api:8000"
TFIDF_SERVICE = "http://tfidf-api:5000"


# ============================================
# Logging
# ============================================
def get_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


log = get_logger()


def debug_enabled() -> bool:
    return os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "on")


def safe_traceback() -> str:
    return "".join(traceback.format_exc())


def shorten(value: str, limit: int = 700) -> str:
    return value if len(value) <= limit else value[:limit] + "… [truncated]"


# ============================================
# Models
# ============================================
class TicketRequest(BaseModel):
    text: str
    return_probas: bool = True
    force_model: Optional[str] = None


class AgentResponse(BaseModel):
    original_text: str
    scrubbed_text: str
    category: str
    confidence: float
    model_used: str
    reasoning: str
    pii_detected: Dict[str, int]
    analysis: Dict[str, Any]
    prediction_time_ms: float


# ============================================
# PII Scrubbing
# ============================================
def scrub_pii(text: str) -> Tuple[str, Dict[str, int]]:
    pii_counts = {"emails": 0, "phones": 0, "credit_cards": 0, "ssn": 0}
    scrubbed = text

    # Emails
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    emails = re.findall(email_pattern, scrubbed)
    pii_counts["emails"] = len(emails)
    scrubbed = re.sub(email_pattern, "[EMAIL]", scrubbed)

    # Phones
    phone_patterns = [
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b",
        r"\b\d{10,}\b",
    ]
    for pattern in phone_patterns:
        matches = re.findall(pattern, scrubbed)
        pii_counts["phones"] += len(matches)
        scrubbed = re.sub(pattern, "[PHONE]", scrubbed)

    # Credit cards
    cc_pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    cards = re.findall(cc_pattern, scrubbed)
    pii_counts["credit_cards"] = len(cards)
    scrubbed = re.sub(cc_pattern, "[CREDIT_CARD]", scrubbed)

    # SSN
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    ssn = re.findall(ssn_pattern, scrubbed)
    pii_counts["ssn"] = len(ssn)
    scrubbed = re.sub(ssn_pattern, "[SSN]", scrubbed)

    return scrubbed, pii_counts


# ============================================
# Text Analysis + Routing Logic
# ============================================
def analyze_text(text: str) -> Dict[str, Any]:
    words = text.split()
    analysis = {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_length": (sum(len(w) for w in words) / len(words)) if words else 0,
        "has_special_chars": bool(re.search(r"[^\x00-\x7F]", text)),
        "has_numbers": bool(re.search(r"\d", text)),
        "uppercase_ratio": (
            (sum(1 for c in text if c.isupper()) / len(text)) if text else 0
        ),
        "sentence_count": len([s for s in re.split(r"[.!?]+", text) if s.strip()]),
    }

    analysis["language"] = (
        "multilingual" if analysis["has_special_chars"] else "english"
    )

    return analysis


def choose_model(text: str, analysis: Dict[str, Any], force: Optional[str]):
    if force:
        if force == "transformer":
            return TRANSFORMER_SERVICE, "Model forced to Transformer"
        if force == "tfidf":
            return TFIDF_SERVICE, "Model forced to TF-IDF"

    if analysis["word_count"] < 5:
        return TFIDF_SERVICE, "Short text → TF-IDF"

    if analysis["language"] == "multilingual":
        return TRANSFORMER_SERVICE, "Multilingual → Transformer"

    if analysis["word_count"] > 50:
        return TRANSFORMER_SERVICE, "Long complex text → Transformer"

    if analysis["sentence_count"] > 2:
        return TRANSFORMER_SERVICE, "Multiple sentences → Transformer"

    return TFIDF_SERVICE, "Standard short/medium text → TF-IDF"


# ============================================
# Metrics Endpoint
# ============================================
@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ============================================
# Root + Health
# ============================================
@app.get("/")
def root():
    return {"service": "agent", "status": "healthy"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================
# Prediction Endpoint
# ============================================
@app.post("/predict", response_model=AgentResponse)
async def predict(request: TicketRequest):

    AGENT_REQUEST_COUNT.labels(status="started").inc()
    start = asyncio.get_event_loop().time()

    # Scrub PII
    scrubbed_text, pii_counts = scrub_pii(request.text)

    for pii_type, count in pii_counts.items():
        if count > 0:
            AGENT_PII_DETECTED.labels(pii_type=pii_type).inc(count)

    # Analyze
    analysis = analyze_text(scrubbed_text)

    # Choose model
    service_url, reasoning = choose_model(scrubbed_text, analysis, request.force_model)

    AGENT_ROUTING_DECISIONS.labels(
        model="transformer" if "transformer" in service_url else "tfidf"
    ).inc()

    # Upstream request
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{service_url}/predict",
                json={"text": scrubbed_text, "return_probas": request.return_probas},
            )
        result = resp.json()

    except Exception:
        AGENT_REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(503, "Model service unreachable")

    # Normalize result
    category = (
        result.get("category")
        or result.get("label")
        or result.get("class")
        or "unknown"
    )

    confidence = result.get("confidence", 0.0)
    model_used = result.get("model", "unknown")

    # Latency
    end = asyncio.get_event_loop().time()
    latency = end - start
    AGENT_LATENCY.observe(latency)
    AGENT_REQUEST_COUNT.labels(status="success").inc()

    return AgentResponse(
        original_text=request.text,
        scrubbed_text=scrubbed_text,
        category=category,
        confidence=float(confidence),
        model_used=model_used,
        reasoning=reasoning,
        pii_detected=pii_counts,
        analysis=analysis,
        prediction_time_ms=round(latency * 1000, 2),
    )


# ============================================
# Run the server
# ============================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6000)
