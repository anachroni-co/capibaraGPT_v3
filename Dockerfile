FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CAPIBARA_STRICT_RUNTIME=1 \
    CAPIBARA_API_HOST=0.0.0.0 \
    CAPIBARA_API_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-mvp.txt /app/requirements-mvp.txt
RUN pip install --no-cache-dir -r /app/requirements-mvp.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1

CMD ["python", "-m", "uvicorn", "capibara.mvp_api:app", "--host", "0.0.0.0", "--port", "8000"]
