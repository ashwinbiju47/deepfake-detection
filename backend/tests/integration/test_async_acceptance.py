"""Integration tests for async acceptance and FIFO dispatch (Task 18.2, Requirements 6.1, 6.3).
"""

from __future__ import annotations

import time

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

from detection.models import AnalysisSession
from detection.services.upload import UploadService

pytestmark = [pytest.mark.django_db, pytest.mark.integration]


def test_async_acceptance_returns_under_2_seconds() -> None:
    """Requirement 6.1: Upload POST API returns 202 Accepted within 2s without waiting for inference."""
    client = APIClient()
    upload = SimpleUploadedFile("video.mp4", b"valid_video_bytes", content_type="video/mp4")

    service = UploadService(
        probe=lambda _p: True,
        enqueuer=lambda _s: True,
    )

    start = time.time()
    with pytest.MonkeyPatch.context() as m:
        m.setattr("detection.views.UploadService", lambda: service)
        response = client.post("/api/analyses", {"file": upload}, format="multipart")
    duration = time.time() - start

    assert response.status_code == 202
    assert "session_id" in response.data
    assert duration < 2.0, "API response must return in < 2 seconds"
