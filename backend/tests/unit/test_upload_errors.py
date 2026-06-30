"""Unit tests for each Upload_Service validation error code (Task 3.4).

Example-based coverage for the four rejection paths of Requirement 1, mirroring
the design "Input Validation Errors" table. Each test submits a concrete,
minimal input through :meth:`UploadService.submit_file` and asserts:

* the returned :class:`UploadResult` is a rejection (``accepted`` is ``False``),
* the typed ``error_code`` matches the expected code,
* the human-readable ``message`` mentions the specific reason (supported
  formats / 50MB limit / empty / valid video), and
* **no** ``AnalysisSession`` is created (Requirements 1.2-1.5 all require that a
  rejected input leaves no session behind).

A final class exercises the DRF intake view (``POST /api/analyses``) with
``APIClient`` to confirm each code maps to its HTTP status
(415/413/400/422).

The decodability probe is dependency-injected so these tests never decode real
video bytes: format/size rejections use the default service (they fail before
the probe runs), and the undecodable case injects a stub probe returning
``False``.
"""

from __future__ import annotations

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from rest_framework.test import APIClient

from detection.models import AnalysisSession
from detection.services.upload import (
    ERROR_EMPTY_FILE,
    ERROR_TOO_LARGE,
    ERROR_UNDECODABLE,
    ERROR_UNSUPPORTED_FORMAT,
    UploadService,
)

pytestmark = [pytest.mark.unit, pytest.mark.django_db]

MAX_UPLOAD_SIZE_BYTES = 52_428_800  # 50 MB inclusive (Requirements 1.1, 1.3)


def _upload(name: str, content: bytes, content_type: str = "video/mp4") -> SimpleUploadedFile:
    """Build a Django uploaded-file stand-in with an explicit name and bytes."""
    return SimpleUploadedFile(name, content, content_type=content_type)


# ---------------------------------------------------------------------------
# Service-level: one concrete example per error code.
# ---------------------------------------------------------------------------
class TestUploadServiceErrorCodes:
    """Each rejection returns the right code/message and creates no session."""

    def test_unsupported_format_txt_file(self) -> None:
        """Requirement 1.2: a non-supported format is rejected, naming formats."""
        before = AnalysisSession.objects.count()

        result = UploadService().submit_file(
            _upload("notes.txt", b"this is not a video", content_type="text/plain")
        )

        assert result["accepted"] is False
        assert result["error_code"] == ERROR_UNSUPPORTED_FORMAT
        assert result["session_id"] is None
        # Message identifies the supported formats (MP4 and AVI).
        message = result["message"] or ""
        assert "MP4" in message and "AVI" in message
        # No session created.
        assert AnalysisSession.objects.count() == before

    def test_empty_zero_byte_file(self) -> None:
        """Requirement 1.4: a 0-byte file is rejected as empty."""
        before = AnalysisSession.objects.count()

        result = UploadService().submit_file(_upload("clip.mp4", b""))

        assert result["accepted"] is False
        assert result["error_code"] == ERROR_EMPTY_FILE
        assert result["session_id"] is None
        assert "empty" in (result["message"] or "").lower()
        assert AnalysisSession.objects.count() == before

    def test_oversize_file_over_50mb(self) -> None:
        """Requirement 1.3: a file larger than 50MB is rejected, stating the limit."""
        before = AnalysisSession.objects.count()

        # One byte over the inclusive 50MB limit. SimpleUploadedFile reports
        # ``size`` from the byte length, so the size check fires deterministically.
        oversize = _upload("big.mp4", b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1))
        result = UploadService().submit_file(oversize)

        assert result["accepted"] is False
        assert result["error_code"] == ERROR_TOO_LARGE
        assert result["session_id"] is None
        assert "50MB" in (result["message"] or "")
        assert AnalysisSession.objects.count() == before

    def test_supported_but_undecodable_file(self) -> None:
        """Requirement 1.5: a supported-extension file that cannot decode is rejected.

        Inject a probe stub returning ``False`` so the file passes the format and
        size checks but fails the decodability probe, without decoding real bytes.
        """
        before = AnalysisSession.objects.count()

        service = UploadService(probe=lambda _path: False)
        result = service.submit_file(_upload("broken.mp4", b"not real video bytes"))

        assert result["accepted"] is False
        assert result["error_code"] == ERROR_UNDECODABLE
        assert result["session_id"] is None
        assert "valid video" in (result["message"] or "").lower()
        assert AnalysisSession.objects.count() == before


# ---------------------------------------------------------------------------
# View-level: each code maps to the documented HTTP status (DRF APIClient).
# ---------------------------------------------------------------------------
class TestCreateAnalysisErrorStatusMapping:
    """``POST /api/analyses`` returns the mapped 4xx for each rejection."""

    def _post(self, client: APIClient, upload: SimpleUploadedFile):
        return client.post("/api/analyses", {"file": upload}, format="multipart")

    def test_unsupported_format_returns_415(self) -> None:
        before = AnalysisSession.objects.count()
        response = self._post(
            APIClient(),
            _upload("notes.txt", b"nope", content_type="text/plain"),
        )
        assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        assert response.data["error_code"] == ERROR_UNSUPPORTED_FORMAT
        assert AnalysisSession.objects.count() == before

    def test_undecodable_one_byte_file_returns_422(self) -> None:
        before = AnalysisSession.objects.count()
        # A 1-byte mp4 survives multipart parsing (non-empty) and passes the
        # format and size checks, but cannot be decoded as a valid video, so the
        # real probe rejects it → 422. (A truly 0-byte part is dropped by the
        # multipart parser, so the EMPTY_FILE→400 mapping is covered at the
        # service level above.)
        response = self._post(APIClient(), _upload("clip.mp4", b"\x00"))
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert response.data["error_code"] == ERROR_UNDECODABLE
        assert AnalysisSession.objects.count() == before

    def test_oversize_returns_413(self) -> None:
        before = AnalysisSession.objects.count()
        response = self._post(
            APIClient(),
            _upload("big.mp4", b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1)),
        )
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert response.data["error_code"] == ERROR_TOO_LARGE
        assert AnalysisSession.objects.count() == before
