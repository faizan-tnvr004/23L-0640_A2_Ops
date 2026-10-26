# Slim Debian-based Python image (~140 MB) instead of the full image (~1 GB)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# curl is needed for the HEALTHCHECK; create an unprivileged user (UID 1000)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 appgroup && useradd -u 1000 -g appgroup -m appuser

WORKDIR /app

# Dependencies BEFORE source: this layer is rebuilt only when requirements change,
# so code edits reuse the cached (slow) pip layer.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py .
# Fallback copy of the champion model for standalone single-container deployment.
# With Docker Compose, ./models is mounted read-only over this directory instead.
COPY models/ ./models/

RUN mkdir -p /app/data && chown -R appuser:appgroup /app
USER appuser

# No secrets here: DATABASE_URL, SECRET_KEY, API_KEY ... are injected at runtime.
ENV MODEL_PATH=/app/models/champion_model.keras \
    DATABASE_URL=sqlite:////app/data/visionguard.db \
    MAX_CONTENT_LENGTH_MB=5

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:create_app()"]
