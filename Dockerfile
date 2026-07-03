# ============================================================
# Batch Video Proctoring Pipeline — Dockerfile
# Multi-stage Python 3.11 slim image for production deployment.
#
# Build:  docker build -t icpc-proc:latest .
# Run:    docker run -p 8000:8000 --env-file .env icpc-proc:latest
# ============================================================

# ---------- Stage 1: Dependency builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for OpenCV, ffmpeg, and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a separate directory for clean copying
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ---------- Stage 2: Runtime image ----------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only runtime system libraries (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY run.py .

# Create non-root user for security
RUN useradd -m -u 1000 procuser && \
    mkdir -p /app/storage && \
    chown -R procuser:procuser /app

USER procuser

# Expose API port
EXPOSE 8000

# Health check for Docker / ECS
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint:
# - gunicorn with 4 UvicornWorker processes
# - Adjust --workers based on CPU count (2 * CPU + 1 is a common guideline)
CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--access-logfile", "-"]
