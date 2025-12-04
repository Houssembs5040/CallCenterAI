FROM python:3.11-slim

# Faster, quieter pip and longer timeouts
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_PROGRESS_BAR=off

WORKDIR /app

# System deps (optional, but good for SSL/certs and curl diagnostics)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Install deps (cache pip to speed up rebuilds)
COPY requirements-agent.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 300 -r /app/requirements.txt

# Copy code
COPY . /app

# Expose + run
EXPOSE 6000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6000"]
