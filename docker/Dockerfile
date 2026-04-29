# Capibara Slim — production image
#
# Build:  docker build -t capibara-slim .
# Run:    docker run -p 8000:8000 capibara-slim

# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements-mvp.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-mvp.txt && \
    pip install --no-cache-dir --prefix=/install pyyaml

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user
RUN useradd -m -u 1000 capibara
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=capibara:capibara . .

USER capibara

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--no-access-log"]
