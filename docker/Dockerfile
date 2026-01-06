# Use a pinned, slim version for security and size
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Install system dependencies (gcc usually needed for some python builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
ARG REQUIREMENTS_FILE=requirements-docker-cpu.txt
COPY requirements.txt requirements.lock requirements-ci.txt requirements-docker-cpu.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/${REQUIREMENTS_FILE}

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Default command (can be overridden)
CMD ["python", "-m", "priorauth.apps.agent.batch_runner"]
