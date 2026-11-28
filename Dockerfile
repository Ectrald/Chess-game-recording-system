# ==============================================================================
# Autonomous Chess Game Recording System - Dockerfile
# Multi-platform deployment (Linux, Windows WSL2, macOS)
# ==============================================================================

FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/games:${PATH}" \
    STOCKFISH_PATH="/usr/games/stockfish"

# Install system dependencies:
# - stockfish: UCI chess engine for move analysis
# - libgl1, libegl1, libglib2.0-0, libgomp1: OpenCV GUI / MediaPipe / image processing dependencies
# - ffmpeg: video codec support
RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    ffmpeg \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source tree, configurations, and ground-truth data
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
COPY main.py pytest.ini /app/

# Ensure output and weights directories exist
RUN mkdir -p /app/data/sessions /app/models /app/data/weights

# Default CLI entrypoint
ENTRYPOINT ["python", "main.py"]
CMD ["record", "--help"]
