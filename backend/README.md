# Deepfake Detection Platform — Backend

Django + DRF + Channels (ASGI) backend with Celery workers on a Redis broker and
PostgreSQL persistence. This is the project skeleton (Task 1.1); pipeline
components are added in later tasks.

## Layout

```
backend/
  config/            # Django project package
    settings.py      # env-driven settings (DRF, Channels, Celery, Redis, flags)
    env.py           # typed, range-clamped env helpers
    celery.py        # Celery app (worker concurrency 1-64)
    asgi.py          # ASGI entry (HTTP + WebSocket routing)
    wsgi.py          # WSGI entry (sync fallback)
    urls.py          # root URLConf (/api/ -> detection)
  detection/         # the analysis app
    apps.py
    models.py        # (models added in Task 2.1)
    views.py         # health endpoint (Upload_Service added in Task 3.x)
    urls.py
    consumers.py     # Result_Streamer WebSocket consumer
    routing.py       # Channels websocket routes
    tasks.py         # Celery tasks (orchestrator added in Task 9.1)
  manage.py
  requirements.txt   # pinned dependencies
  .env.example       # configuration template
  pytest.ini         # pytest + pytest-django config
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit as needed
python manage.py migrate
python manage.py check
```

## Running

```bash
# ASGI server (HTTP + WebSocket)
python manage.py runserver           # daphne-backed in DEBUG

# Celery worker (concurrency from CELERY_WORKER_CONCURRENCY, 1-64)
celery -A config worker -l info --concurrency "$CELERY_WORKER_CONCURRENCY"
```

## Configuration

All settings read from environment variables with sensible defaults. Key knobs:

| Variable | Default | Notes |
|----------|---------|-------|
| `CELERY_WORKER_CONCURRENCY` | 4 | Clamped to 1-64 (Req 6.4) |
| `DECISION_THRESHOLD` | 0.5 | Clamped to [0,1] (Req 4.3) |
| `FUSION_WEIGHT_VISUAL` / `FUSION_WEIGHT_AUDIO` | 0.6 / 0.4 | Fusion weights (Req 4.1) |
| `MAX_UPLOAD_SIZE_BYTES` | 52428800 | 50 MB limit (Req 1) |
| `EXTERNAL_URL_ENABLED` | false | Should-Have gate (Req 10) |
| `HEATMAP_ENABLED` | false | Should-Have gate (Req 11) |
| `REPORT_ENABLED` | false | Could-Have gate (Req 12) |

Uploaded media is transient: stored under `MEDIA_ROOT/<session_id>/` and purged
after analysis (Req 9).
