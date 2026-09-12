---
title: Deepfake Detection Platform
emoji: 🕵️
colorFrom: indigo
colorTo: sky
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Real-Time Deepfake Detection Platform

Multi-modal (vision + audio) deepfake detection with explainable AI (Grad-CAM
ORIGINAL / HEATMAP / OVERLAY triple), live WebSocket progress streaming, PDF
reports, and a Results/Evaluation chapter comparing **Multimodal >
Visual-only > Audio-only** with cross-dataset generalization.

- **Frontend:** React + Vite + Tailwind (built at image build time)
- **Backend:** Django + DRF + Channels/Daphne (HTTP + WebSocket on :7860)
- **Queue:** Celery worker + in-container Redis broker/channel layer
- **DB:** SQLite (single-container; evaluation data is reference material)
- **ML:** CPU PyTorch, OpenCV, librosa, MTCNN (TensorFlow 2.15 CPU)

## Deploying

This Space builds the whole platform from the `Dockerfile` at the repo root.
Set these variables in **Settings → Variables and secrets**:

| Variable | Value | Purpose |
|---|---|---|
| `DJANGO_SECRET_KEY` | a long random string | Django signing (required) |
| `DJANGO_DEBUG` | `false` | production mode |
| `DJANGO_ALLOWED_HOSTS` | `*` | accept `.hf.space` hostnames |
| `DJANGO_DB_ENGINE` | `sqlite` | embedded DB, no external service |
| `DJANGO_SERVE_SPA` | `true` | serve the built React app same-origin |
| `EXTERNAL_URL_ENABLED` | `true` | allow analyzing videos by URL |
| `HEATMAP_ENABLED` | `true` | Grad-CAM heatmap + overlay panels |
| `REPORT_ENABLED` | `true` | downloadable PDF reports |
| `CELERY_WORKER_CONCURRENCY` | `2` | worker capacity (2 vCPU) |
| `MAX_UPLOAD_SIZE_BYTES` | `30000000` | 30 MB cap — the Spaces proxy rejects bodies near 50 MB |

See `DEPLOY.md` in the repo for the full step-by-step checklist.

## Usage

Upload a video (mp4/avi, up to 50 MB) or paste a direct video URL, then watch
the pipeline run live: frame extraction → face detection → visual model →
audio extraction → audio model → multimodal fusion → fake probability →
XAI explanation → final result.