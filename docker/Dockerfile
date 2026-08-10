# ── GCS image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS gcs

WORKDIR /app

# System deps for Redis client hiredis
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    gcc libhiredis-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY python/ ./python/
COPY config/  ./config/
COPY proto/   ./proto/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "python.gcs.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ── Celery worker image (same base, different entrypoint) ─────────────────────
FROM gcs AS celery-worker

CMD ["celery", "-A", "python.gcs.workers.app", "worker", \
     "--loglevel=info", "--concurrency=4", "-Q", "cellhawk"]
