"""Celery application for the Deepfake Detection Platform.

The worker capacity is driven by ``CELERY_WORKER_CONCURRENCY`` (1-64,
Requirement 6.4) and all broker/result settings come from Django settings via
the ``CELERY_`` namespace.
"""

from __future__ import annotations

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("deepfake_detection")

# Pull configuration from Django settings using the CELERY_ prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks.py modules in all installed apps (e.g. detection.tasks).
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self) -> None:  # pragma: no cover - diagnostic helper
    print(f"Request: {self.request!r}")
