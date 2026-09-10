# syntax=docker/dockerfile:1
#
# Deepfake Detection Platform — single-container image for Hugging Face Spaces.
#
# Stage 1 builds the React frontend (Vite), Stage 2 assembles the Python
# runtime: Daphne (ASGI: HTTP + WebSocket) + Celery worker + Redis broker,
# all supervised in one process tree. The built SPA is served same-origin by
# Django/WhiteNoise so the frontend talks to /api and /ws with no CORS.

# ---- Stage 1: build the React frontend -----------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: backend runtime ---------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies:
#   ffmpeg            -> video/audio decoding (opencv, ffmpeg-python, librosa)
#   redis-server      -> Celery broker + Channels channel layer (in-container)
#   pango/cairo/gdk   -> WeasyPrint HTML->PDF rendering
#   libgl1/libglib2.0 -> OpenCV runtime
#   fonts-dejavu-core -> PDF text glyphs
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        redis-server \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi-dev \
        shared-mime-info \
        fonts-dejavu-core \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/ /app/backend/
COPY --from=frontend-build /build/frontend/dist /app/frontend/dist

# CPU-only PyTorch first (the free tier has no GPU; the PyPI default wheel
# bundles CUDA and would add ~2.5 GB for nothing), then the pinned requirements
# (torch/torchvision are already satisfied and skipped), then TensorFlow CPU
# (mtcnn 0.1.1 hard-imports tensorflow.keras; 2.15.x is the last line whose
# tf.keras is Keras 2, which mtcnn is compatible with).
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/backend/requirements.txt \
    && pip install --no-cache-dir "tensorflow-cpu==2.15.1"

COPY supervisord.conf /etc/supervisord.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860
WORKDIR /app/backend
ENTRYPOINT ["/entrypoint.sh"]