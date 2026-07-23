"""Unit tests for URL intake edge cases (Task 14.3, Requirements 10.1, 10.2, 10.4, 10.5, 10.6).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from detection.models import AnalysisSession
from detection.services.upload import UploadService

pytestmark = [pytest.mark.django_db, pytest.mark.unit]


def test_url_input_disabled_rejection() -> None:
    """Requirement 10.1: submit_url rejected when EXTERNAL_URL_ENABLED is False."""
    service = UploadService()
    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.EXTERNAL_URL_ENABLED", False)
        res = service.submit_url("https://example.com/video.mp4")

        assert res["accepted"] is False
        assert res["error_code"] == "URL_INPUT_DISABLED"


def test_url_input_success_path() -> None:
    """Requirement 10.1 & 10.2: Valid URL creates session and enqueues task."""
    def fake_fetcher(_url: str, dest: Path) -> bool:
        dest.write_bytes(b"valid_video_bytes")
        return True

    service = UploadService(
        probe=lambda _p: True,
        enqueuer=lambda _s: True,
    )

    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.EXTERNAL_URL_ENABLED", True)
        res = service.submit_url("https://example.com/video.mp4", fetcher=fake_fetcher)

        assert res["accepted"] is True
        assert res["session_id"] is not None

        session = AnalysisSession.objects.get(id=res["session_id"])
        assert session.source_type == AnalysisSession.SourceType.URL
        assert session.source_ref == "https://example.com/video.mp4"


def test_url_input_unreachable_error() -> None:
    """Requirement 10.4: Unreachable URL yields URL_UNREACHABLE error."""
    def failing_fetcher(_url: str, _dest: Path) -> bool:
        raise OSError("Connection timed out")

    service = UploadService()
    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.EXTERNAL_URL_ENABLED", True)
        res = service.submit_url("https://unreachable.example.com/video.mp4", fetcher=failing_fetcher)

        assert res["accepted"] is False
        assert res["error_code"] == "URL_UNREACHABLE"


def test_url_input_oversize_error() -> None:
    """Requirement 10.6: Oversize URL stream yields TOO_LARGE error."""
    def oversize_fetcher(_url: str, _dest: Path) -> bool:
        raise ValueError("TOO_LARGE")

    service = UploadService()
    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.EXTERNAL_URL_ENABLED", True)
        res = service.submit_url("https://huge.example.com/large.mp4", fetcher=oversize_fetcher)

        assert res["accepted"] is False
        assert res["error_code"] == "TOO_LARGE"
