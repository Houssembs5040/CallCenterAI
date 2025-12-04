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
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
# Service URLs for Docker network (container names)
TRANSFORMER_SERVICE = "http://transformer-api:8000"
TFIDF_SERVICE = "http://tfidf-api:5000"

# Fallback to localhost if not in Docker network (for testing)
TRANSFORMER_SERVICE_LOCAL = "http://transformer-api:8000"
TFIDF_SERVICE_LOCAL = "http://tfidf-api:5000"


# ============================================
# Logging & Debug Helpers
# ============================================
def get_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(level)
        handler = logging.StreamHandler()
        # Simple JSON-ish line logging for easy grep/ship
        formatter = logging.Formatter(
            '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


log = get_logger()


def debug_enabled() -> bool:
    # Set DEBUG=true to include stack traces and upstream body previews in error payloads
    return os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "on")


def safe_traceback() -> str:
    # Full traceback for logs; optionally returned in response if DEBUG=true
    return "".join(traceback.format_exc())


def shorten(value: str, limit: int = 700) -> str:
    if value is None:
        return ""
    return value if len(value) <= limit else value[:limit] + "… [truncated]"


# ============================================
# Request/Response Models
# ============================================
class TicketRequest(BaseModel):
    """Incoming ticket request"""

    text: str
    return_probas: bool = True
    force_model: Optional[str] = (
        None  # "transformer" or "tfidf" to force a specific model
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "My laptop screen is broken, contact me at john@email.com or 555-1234",
                "return_probas": True,
                "force_model": None,
            }
        }


class AgentResponse(BaseModel):
    """Agent response with routing decision"""

    original_text: str
    scrubbed_text: str
    category: str
    confidence: float
    model_used: str
    reasoning: str
    pii_detected: Dict[str, int]
    analysis: Dict[str, Any]
    prediction_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "My laptop screen is broken, contact me at john@email.com",
                "scrubbed_text": "My laptop screen is broken, contact me at [EMAIL]",
                "category": "Hardware",
                "confidence": 0.98,
                "model_used": "transformer",
                "reasoning": "Complex hardware issue - using Transformer for better accuracy",
                "pii_detected": {"emails": 1, "phones": 0},
                "analysis": {
                    "word_count": 8,
                    "char_count": 45,
                    "avg_word_length": 5.6,
                    "has_special_chars": False,
                    "has_numbers": True,
                    "uppercase_ratio": 0.0,
                    "sentence_count": 1,
                    "language": "english",
                },
                "prediction_time_ms": 125.5,
            }
        }


# ============================================
# PII Scrubbing Functions
# ============================================
def scrub_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Remove Personally Identifiable Information (PII)
    Returns: (scrubbed_text, pii_counts)
    """
    pii_counts = {"emails": 0, "phones": 0, "credit_cards": 0, "ssn": 0}

    scrubbed = text

    # Remove emails
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails_found = re.findall(email_pattern, scrubbed)
    pii_counts["emails"] = len(emails_found)
    scrubbed = re.sub(email_pattern, "[EMAIL]", scrubbed)

    # Remove phone numbers (various formats)
    phone_patterns = [
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # 555-123-4567, 5551234567
        r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b",  # (555) 123-4567
        r"\b\d{10,}\b",  # 10+ digit numbers
    ]
    for pattern in phone_patterns:
        phones_found = re.findall(pattern, scrubbed)
        pii_counts["phones"] += len(phones_found)
        scrubbed = re.sub(pattern, "[PHONE]", scrubbed)

    # Remove credit card numbers (basic pattern)
    cc_pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    cc_found = re.findall(cc_pattern, scrubbed)
    pii_counts["credit_cards"] = len(cc_found)
    scrubbed = re.sub(cc_pattern, "[CREDIT_CARD]", scrubbed)

    # Remove SSN (xxx-xx-xxxx)
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    ssn_found = re.findall(ssn_pattern, scrubbed)
    pii_counts["ssn"] = len(ssn_found)
    scrubbed = re.sub(ssn_pattern, "[SSN]", scrubbed)

    return scrubbed, pii_counts


# ============================================
# Text Analysis Functions
# ============================================
def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text characteristics"""
    words = text.split()

    analysis = {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_length": (
            (sum(len(word) for word in words) / len(words)) if words else 0
        ),
        "has_special_chars": bool(re.search(r"[^\x00-\x7F]", text)),  # Non-ASCII chars
        "has_numbers": bool(re.search(r"\d", text)),
        "uppercase_ratio": (
            (sum(1 for c in text if c.isupper()) / len(text)) if text else 0
        ),
        "sentence_count": len([s for s in re.split(r"[.!?]+", text) if s.strip()]),
    }

    # Simple language detection (very basic)
    if re.search(r"[^\x00-\x7F]", text):
        analysis["language"] = "multilingual"
    else:
        analysis["language"] = "english"

    return analysis


def choose_model(
    text: str, analysis: Dict[str, Any], force_model: Optional[str] = None
) -> Tuple[str, str]:
    """
    Choose the best model based on text analysis
    Returns: (service_url, reasoning)
    """
    # If model is forced, use it
    if force_model:
        if force_model.lower() == "transformer":
            return TRANSFORMER_SERVICE_LOCAL, "Model forced to Transformer by user"
        elif force_model.lower() == "tfidf":
            return TFIDF_SERVICE_LOCAL, "Model forced to TF-IDF by user"

    # Decision logic based on analysis

    # Rule 1: Very short text (< 5 words) → TF-IDF (fast)
    if analysis["word_count"] < 5:
        return (
            TFIDF_SERVICE_LOCAL,
            "Very short text - using fast TF-IDF model for efficiency",
        )

    # Rule 2: Multilingual or special characters → Transformer
    if analysis["has_special_chars"] or analysis["language"] == "multilingual":
        return (
            TRANSFORMER_SERVICE_LOCAL,
            "Multilingual text detected - using Transformer for better language support",
        )

    # Rule 3: Long complex text (> 50 words) → Transformer
    if analysis["word_count"] > 50:
        return (
            TRANSFORMER_SERVICE_LOCAL,
            "Long complex text - using Transformer for better context understanding",
        )

    # Rule 4: High uppercase ratio (might be urgent) → Transformer
    if analysis["uppercase_ratio"] > 0.5:
        return (
            TRANSFORMER_SERVICE_LOCAL,
            "High urgency detected (uppercase text) - using Transformer for accurate classification",
        )

    # Rule 5: Multiple sentences → Transformer
    if analysis["sentence_count"] > 2:
        return (
            TRANSFORMER_SERVICE_LOCAL,
            "Multiple sentences detected - using Transformer for better context",
        )

    # Rule 6: Medium length text (5-20 words) → TF-IDF (faster, good enough)
    if 5 <= analysis["word_count"] <= 20:
        return TFIDF_SERVICE_LOCAL, "Medium-length simple text - using TF-IDF for speed"

    # Default: Use Transformer for complex cases
    return (
        TRANSFORMER_SERVICE_LOCAL,
        "Standard ticket - using Transformer for highest accuracy",
    )


# ============================================
# Middleware: request id, timing, error logging
# ============================================
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = asyncio.get_event_loop().time()

    try:
        response = await call_next(request)
        duration_ms = (asyncio.get_event_loop().time() - start) * 1000
        log.info(
            json.dumps(
                {
                    "event": "request_completed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            )
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        duration_ms = (asyncio.get_event_loop().time() - start) * 1000
        log.error(
            json.dumps(
                {
                    "event": "unhandled_exception",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "trace": safe_traceback(),
                }
            )
        )
        raise


# ============================================
# Global Exception Handlers
# ============================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    log.warning(
        json.dumps(
            {
                "event": "validation_error",
                "request_id": request_id,
                "errors": exc.errors(),
            }
        )
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request payload.",
                "details": exc.errors(),
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(httpx.HTTPError)
async def httpx_exception_handler(request: Request, exc: httpx.HTTPError):
    request_id = getattr(request.state, "request_id", "unknown")
    log.error(
        json.dumps(
            {
                "event": "httpx_error",
                "request_id": request_id,
                "type": exc.__class__.__name__,
                "trace": safe_traceback(),
            }
        )
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "UPSTREAM_HTTP_ERROR",
                "message": "Failed to reach the model service.",
                "request_id": request_id,
                **({"trace": safe_traceback()} if debug_enabled() else {}),
            }
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    log.error(
        json.dumps(
            {
                "event": "unhandled_exception_response",
                "request_id": request_id,
                "type": exc.__class__.__name__,
                "trace": safe_traceback(),
            }
        )
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred.",
                "request_id": request_id,
                **({"trace": safe_traceback()} if debug_enabled() else {}),
            }
        },
    )


# ============================================
# API Endpoints
# ============================================
@app.get("/")
def root():
    """Service information"""
    return {
        "service": "🤖 AI Agent Service",
        "status": "✅ healthy",
        "description": "Intelligent routing to best classification model",
        "features": [
            "PII Scrubbing (emails, phones, credit cards, SSN)",
            "Text Analysis",
            "Intelligent Model Selection",
            "Routing to TF-IDF or Transformer",
            "Detailed Reasoning",
            "Rich, PII-safe debugging with correlation IDs",
        ],
        "available_models": {
            "transformer": TRANSFORMER_SERVICE_LOCAL,
            "tfidf": TFIDF_SERVICE_LOCAL,
        },
    }


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "✅ healthy",
        "service": "agent",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=AgentResponse)
async def predict(request: TicketRequest, fastapi_request: Request):
    """
    Intelligent ticket classification with routing

    1. Scrubs PII from text
    2. Analyzes text characteristics
    3. Chooses best model
    4. Routes to selected model
    5. Returns prediction with reasoning
    """
    start_time = asyncio.get_event_loop().time()
    request_id = getattr(fastapi_request.state, "request_id", str(uuid.uuid4()))

    # Validate input
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Step 1: Scrub PII
    scrubbed_text, pii_counts = scrub_pii(request.text)

    # Step 2: Analyze text
    analysis = analyze_text(scrubbed_text)

    # Step 3: Choose model
    service_url, reasoning = choose_model(scrubbed_text, analysis, request.force_model)

    # Step 4: Make prediction
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            upstream_payload = {
                "text": scrubbed_text,
                "return_probas": request.return_probas,
            }
            log.info(
                json.dumps(
                    {
                        "event": "upstream_request",
                        "request_id": request_id,
                        "service_url": f"{service_url}/predict",
                        "payload_preview": {
                            "return_probas": request.return_probas,
                            "text_len": len(scrubbed_text),
                        },
                    }
                )
            )

            response = await client.post(
                f"{service_url}/predict", json=upstream_payload
            )

            # Capture body safely for debugging (do not log original text)
            resp_text = ""
            try:
                resp_bytes = await response.aread()
                resp_text = resp_bytes.decode(errors="replace")
            except Exception:
                resp_text = "<unreadable response body>"

            if response.status_code >= 400:
                # Log detailed upstream failure with truncated body
                log.error(
                    json.dumps(
                        {
                            "event": "upstream_error",
                            "request_id": request_id,
                            "service_url": str(response.request.url),
                            "status_code": response.status_code,
                            "response_body": shorten(resp_text, 1200),
                        }
                    )
                )
                # Map common categories
                status_map = {
                    408: 504,
                    429: 502,
                    500: 502,
                    502: 502,
                    503: 503,
                    504: 504,
                }
                mapped = status_map.get(response.status_code, 502)
                raise HTTPException(
                    status_code=mapped,
                    detail={
                        "code": "MODEL_SERVICE_ERROR",
                        "message": "Model service returned an error.",
                        "upstream_status": response.status_code,
                        "service_url": str(response.request.url),
                        "request_id": request_id,
                        "upstream_body_preview": (
                            shorten(resp_text, 500) if debug_enabled() else None
                        ),
                    },
                )

            # Parse JSON safely
            try:
                result = json.loads(resp_text)
            except json.JSONDecodeError:
                log.error(
                    json.dumps(
                        {
                            "event": "upstream_invalid_json",
                            "request_id": request_id,
                            "service_url": str(response.request.url),
                            "response_body": shorten(resp_text, 1200),
                        }
                    )
                )
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "MODEL_INVALID_RESPONSE",
                        "message": "Model service returned invalid JSON.",
                        "service_url": str(response.request.url),
                        "request_id": request_id,
                    },
                )

        # Calculate prediction time
        end_time = asyncio.get_event_loop().time()
        prediction_time_ms = (end_time - start_time) * 1000

        # ---------- Normalize upstream payload safely ----------
        # Try common key variants for category
        category = (
            result.get("category")
            or result.get("label")
            or result.get("class")
            or "unknown"
        )

        # Confidence can be absent or null; try alternatives
        confidence = result.get("confidence")

        # If none, try to derive from probas-like structures
        if confidence is None:
            for probs_key in ("probas", "probabilities", "scores"):
                probs = result.get(probs_key)
                if isinstance(probs, dict) and probs:
                    try:
                        # Use the max probability value
                        confidence = float(max(probs.values()))
                        break
                    except Exception:
                        pass

        # Final fallback
        if confidence is None:
            confidence = 0.0

        # Model name (optional upstream)
        model_used = result.get("model", "unknown")

        # Log if upstream was missing expected keys
        missing_keys = []
        if "category" not in result and "label" not in result and "class" not in result:
            missing_keys.append("category/label/class")
        if "confidence" not in result and not any(
            k in result for k in ("probas", "probabilities", "scores")
        ):
            missing_keys.append("confidence/probas")

        if missing_keys:
            log.warning(
                json.dumps(
                    {
                        "event": "upstream_missing_fields",
                        "request_id": request_id,
                        "missing": missing_keys,
                        "derived_confidence": confidence,
                        "category": category,
                    }
                )
            )

        log.info(
            json.dumps(
                {
                    "event": "prediction_success",
                    "request_id": request_id,
                    "model_used": model_used,
                    "category": category,
                    "confidence": confidence,
                    "duration_ms": round(prediction_time_ms, 2),
                }
            )
        )

        # ---------- Build response ----------
        return AgentResponse(
            original_text=request.text,
            scrubbed_text=scrubbed_text,
            category=category,
            confidence=float(confidence),
            model_used=model_used,
            reasoning=reasoning,
            pii_detected=pii_counts,
            analysis=analysis,
            prediction_time_ms=round(prediction_time_ms, 2),
        )

    except httpx.ConnectTimeout:
        log.error(
            json.dumps(
                {
                    "event": "upstream_connect_timeout",
                    "request_id": request_id,
                    "service_url": service_url,
                    "trace": safe_traceback(),
                }
            )
        )
        raise HTTPException(
            status_code=504,
            detail={
                "code": "MODEL_CONNECT_TIMEOUT",
                "message": "Timed out connecting to model service.",
                "service_url": service_url,
                "request_id": request_id,
                **({"trace": safe_traceback()} if debug_enabled() else {}),
            },
        )
    except httpx.ReadTimeout:
        log.error(
            json.dumps(
                {
                    "event": "upstream_read_timeout",
                    "request_id": request_id,
                    "service_url": service_url,
                    "trace": safe_traceback(),
                }
            )
        )
        raise HTTPException(
            status_code=504,
            detail={
                "code": "MODEL_READ_TIMEOUT",
                "message": "Timed out waiting for model response.",
                "service_url": service_url,
                "request_id": request_id,
                **({"trace": safe_traceback()} if debug_enabled() else {}),
            },
        )
    except httpx.RequestError as e:
        # DNS failure, refused connection, etc
        log.error(
            json.dumps(
                {
                    "event": "upstream_request_error",
                    "request_id": request_id,
                    "service_url": service_url,
                    "type": e.__class__.__name__,
                    "trace": safe_traceback(),
                }
            )
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_UNREACHABLE",
                "message": "Failed to reach model service.",
                "service_url": service_url,
                "request_id": request_id,
                **({"trace": safe_traceback()} if debug_enabled() else {}),
            },
        )
    except HTTPException:
        # Already structured and logged above
        raise
    except Exception:
        # Final guardrail for truly unexpected errors
        log.error(
            json.dumps(
                {
                    "event": "predict_unexpected_error",
                    "request_id": request_id,
                    "trace": safe_traceback(),
                }
            )
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Prediction failed due to an unexpected error.",
                "request_id": request_id,
                **({"trace": safe_traceback()} if debug_enabled() else {}),
            },
        )


@app.post("/analyze")
async def analyze_ticket(request: TicketRequest):
    """
    Analyze ticket without making prediction
    Useful for testing routing logic
    """
    # Scrub PII
    scrubbed_text, pii_counts = scrub_pii(request.text)

    # Analyze
    analysis = analyze_text(scrubbed_text)

    # Choose model
    service_url, reasoning = choose_model(scrubbed_text, analysis, request.force_model)

    return {
        "original_text": request.text,
        "scrubbed_text": scrubbed_text,
        "pii_detected": pii_counts,
        "analysis": analysis,
        "recommended_model": "transformer" if "transformer" in service_url else "tfidf",
        "reasoning": reasoning,
    }


# ============================================
# Run the server
# ============================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🤖 Starting AI Agent Service...")
    print("📍 API will be available at: http://localhost:6000")
    print("📚 Documentation at: http://localhost:6000/docs")
    print("=" * 60 + "\n")

    # In dev, you can enable deep error payloads:
    #   export DEBUG=true
    # And adjust log level:
    #   export LOG_LEVEL=DEBUG
    uvicorn.run(
        app, host="0.0.0.0", port=6000, log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
