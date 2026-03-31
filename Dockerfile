# ─────────────────────────────────────────────────────────────────────────────
# National Space Hackathon 2026 — ACM Backend
# Base image: ubuntu:22.04 (required per problem statement §8)
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.10 + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
        libatlas-base-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# Use python3 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/ ./data/

# Expose port 8000 (required per §8)
EXPOSE 8000

# Bind to 0.0.0.0 so the grading scripts can reach the API
CMD ["python3", "-m", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
