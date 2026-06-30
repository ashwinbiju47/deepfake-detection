"""
Django settings for the Real-Time Deepfake Detection Platform.

Configuration is read from environment variables with sensible defaults so the
project runs out-of-the-box in development while remaining 12-factor friendly
in production. See `.env.example` for the full list of supported variables.

Design references:
- Technology Mapping: Django + DRF (API), Channels/ASGI (realtime),
  Celery + Redis (queue/broker + channel layer), PostgreSQL (persistence),
  transient media store keyed by ``session_id``.
- Concurrency Model: ``CELERY_WORKER_CONCURRENCY`` is configurable 1-64
  (Requirement 6.4).
"""

from pathlib import Path

from .env import (
    env_bool,
    env_int_clamped,
    env_float_clamped,
    env_list,
    env_str,
    load_dotenv,
)

# ---------------------------------------------------------------------------
# Paths & .env loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Load variables from a local .env file (if present) before reading settings.
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Core security / debug
# ---------------------------------------------------------------------------
SECRET_KEY = env_str("DJANGO_SECRET_KEY", "django-insecure-dev-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
INSTALLED_APPS = [
    "daphne",  # ASGI server; must precede staticfiles for runserver override
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    "channels",
    # Local
    "detection",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# Both WSGI and ASGI entrypoints are provided; ASGI is primary (Channels).
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ---------------------------------------------------------------------------
# Database (PostgreSQL via the Django ORM)
# ---------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env_str("POSTGRES_DB", "deepfake"),
        "USER": env_str("POSTGRES_USER", "deepfake"),
        "PASSWORD": env_str("POSTGRES_PASSWORD", "deepfake"),
        "HOST": env_str("POSTGRES_HOST", "localhost"),
        "PORT": env_str("POSTGRES_PORT", "5432"),
        "CONN_MAX_AGE": env_int_clamped("POSTGRES_CONN_MAX_AGE", 60, 0, 3600),
    }
}

# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ---------------------------------------------------------------------------
# Internationalization
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
}

# ---------------------------------------------------------------------------
# Redis (shared broker + channel-layer backing store)
# ---------------------------------------------------------------------------
REDIS_URL = env_str("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Channels (ASGI realtime) — Redis channel layer for cross-process fan-out
# ---------------------------------------------------------------------------
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [env_str("CHANNEL_LAYER_REDIS_URL", REDIS_URL)],
        },
    }
}

# ---------------------------------------------------------------------------
# Celery (async task queue) — Redis broker + result backend
# ---------------------------------------------------------------------------
CELERY_BROKER_URL = env_str("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = env_str("CELERY_RESULT_BACKEND", REDIS_URL)
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE

# Reliability settings (design: Task_Queue). Jobs beyond capacity stay queued
# in FIFO order and are never dropped (Requirements 6.3, 6.5).
CELERY_TASK_ACKS_LATE = True
CELERY_TASK_REJECT_ON_WORKER_LOST = True
CELERY_TASK_DEFAULT_QUEUE = "analysis"
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # strict FIFO-ish, one task at a time

# Retry policy (Requirement 6.6): 1 initial attempt + up to 3 retries.
CELERY_TASK_MAX_RETRIES = env_int_clamped("CELERY_TASK_MAX_RETRIES", 3, 0, 10)

# Configurable worker capacity, clamped to the supported range 1-64
# (Requirement 6.4).
CELERY_WORKER_CONCURRENCY = env_int_clamped("CELERY_WORKER_CONCURRENCY", 4, 1, 64)

# ---------------------------------------------------------------------------
# Transient media store (keyed by session_id) — GDPR transient handling (Req 9)
# ---------------------------------------------------------------------------
# Uploaded videos and intermediate artifacts live here under a per-session
# subdirectory and are purged after analysis. No media is stored in the DB.
MEDIA_ROOT = Path(env_str("MEDIA_ROOT", str(BASE_DIR / "transient_media")))
MEDIA_URL = "media/"
# NOTE: the per-session transient path helper lives in
# ``detection.storage.session_media_dir`` (Django only exposes UPPERCASE
# settings, so the helper cannot live here).

# ---------------------------------------------------------------------------
# Feature flags (MoSCoW gating)
# ---------------------------------------------------------------------------
# Should-Have: external URL input (Req 10) and Grad-CAM heatmaps (Req 11).
# Could-Have: downloadable PDF report (Req 12).
EXTERNAL_URL_ENABLED = env_bool("EXTERNAL_URL_ENABLED", False)
HEATMAP_ENABLED = env_bool("HEATMAP_ENABLED", False)
REPORT_ENABLED = env_bool("REPORT_ENABLED", False)

# ---------------------------------------------------------------------------
# Domain configuration: upload limits, decision threshold, fusion weights
# ---------------------------------------------------------------------------
# Supported video container formats (Requirement 1).
SUPPORTED_VIDEO_FORMATS = env_list("SUPPORTED_VIDEO_FORMATS", "mp4,avi")

# Hard upload size limit: 50 MB inclusive (Requirements 1.1, 1.3, 10.6).
MAX_UPLOAD_SIZE_BYTES = env_int_clamped(
    "MAX_UPLOAD_SIZE_BYTES", 52_428_800, 1, 52_428_800
)

# Decision threshold in [0.0, 1.0] (Requirement 4.3): score >= threshold ->
# "deepfake", else "authentic".
DECISION_THRESHOLD = env_float_clamped("DECISION_THRESHOLD", 0.5, 0.0, 1.0)

# Fusion weights for the weighted-combination of modalities (Requirement 4.1).
FUSION_WEIGHT_VISUAL = env_float_clamped("FUSION_WEIGHT_VISUAL", 0.6, 0.0, 1.0)
FUSION_WEIGHT_AUDIO = env_float_clamped("FUSION_WEIGHT_AUDIO", 0.4, 0.0, 1.0)

# External URL retrieval timeout in seconds (Requirement 10.4).
URL_FETCH_TIMEOUT_SECONDS = env_int_clamped("URL_FETCH_TIMEOUT_SECONDS", 30, 1, 300)

# Media purge deadline in seconds after session completion (Requirement 9.1).
MEDIA_PURGE_DEADLINE_SECONDS = env_int_clamped(
    "MEDIA_PURGE_DEADLINE_SECONDS", 60, 1, 600
)
