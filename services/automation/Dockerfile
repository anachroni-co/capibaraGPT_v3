# Capibara6 N8N Automation Service Dockerfile
# ==========================================

FROM python:3.11-slim

LABEL maintainer="Capibara6 Team"
LABEL description="Capibara6 N8N Automation Service - AI-powered workflow automation"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV APP_HOME=/app

# Create app user
RUN groupadd -r capibara && useradd -r -g capibara capibara

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR $APP_HOME

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/temp /app/models && \
    chown -R capibara:capibara /app

# Copy requirements first for better caching
COPY capibara/services/automation/requirements.txt /app/requirements-automation.txt
COPY requirements-enterprise.txt /app/requirements-enterprise.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements-automation.txt && \
    pip install --no-cache-dir -r /app/requirements-enterprise.txt

# Copy the entire project
COPY . /app/

# Set proper permissions
RUN chown -R capibara:capibara /app

# Switch to non-root user
USER capibara

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "capibara/services/automation/main.py", "--host", "0.0.0.0", "--port", "8000"]