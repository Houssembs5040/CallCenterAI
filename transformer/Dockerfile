# ============================================
# Stage 1: Base Image
# ============================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ============================================
# Stage 2: Install Dependencies
# ============================================

# Copy requirements first (for Docker cache optimization)
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# ============================================
# Stage 3: Copy Application Code
# ============================================

# Copy source code
COPY src/ ./src/

# Copy API code
COPY . ./api/transformer_service/

# Copy models (this will be large ~500MB)
COPY model/transformer/ ./models/transformer/

# ============================================
# Stage 4: Configuration
# ============================================

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ============================================
# Stage 5: Run Application
# ============================================

# Run the API
CMD ["python", "api/transformer_service/main.py"]