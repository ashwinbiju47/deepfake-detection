"""Tests for multi-kind intake (video / image / audio) and URL removal.

Covers:

* media-kind detection from the extension (``media_kind_for``);
* acceptance of image and audio uploads with the right ``media_kind`` recorded;
* per-kind decodability probing (an invalid image is rejected UNDECODABLE);
* the URL-submission removal (``url`` payload -> 400 URL_NOT_SUPPORTED);
* the ``GET /api/analyses/{id}`` last-analysis metrics endpoint.
"""

from __future__ import annotations

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from rest_framework.test import APIClient

from detection.models import AnalysisSession
from detection.services.upload import UploadService, media_kind_for

pytestmark = [pytest.mark.django_db, pytest.mark.unit]


# ---------------------------------------------------------------------------
# media_kind_for: extension -> kind classification
# ---------------------------------------------------------------------------


class TestMediaKindDetection:
    def test_video_extensions(self) -> None:
        assert media_kind_for("clip.mp4") == "video"
        assert media_kind_for("clip.AVI") == "video"
        assert media_kind_for("clip.mov") == "video"

    def test_image_extensions(self) -> None:
        assert media_kind_for("face.jpg") == "image"
        assert media_kind_for("face.PNG") == "image"
        assert media_kind_for("photo.webp") == "image"

    def test_audio_extensions(self) -> None:
        assert media_kind_for("speech.wav") == "audio"
        assert media_kind_for("speech.MP3") == "audio"
        assert media_kind_for("voice.flac") == "audio"

    def test_unsupported_and_missing(self) -> None:
        assert media_kind_for("archive.zip") is None
        assert media_kind_for("noext") is None
        assert media_kind_for(None) is None


# ---------------------------------------------------------------------------
# Service-level acceptance of image / audio uploads
# ---------------------------------------------------------------------------


class TestImageAudioAcceptance:
    def test_image_upload_accepted_with_kind(self, tmp_path, settings) -> None:
        settings.MEDIA_ROOT = str(tmp_path)
        upload = SimpleUploadedFile(
            "face.jpg", b"fake-jpeg-bytes", content_type="image/jpeg"
        )
        result = UploadService(probe=lambda _p: True).submit_file(upload)

        assert result["accepted"] is True
        assert result["media_kind"] == "image"
        session = AnalysisSession.objects.get(id=result["session_id"])
        assert session.media_kind == AnalysisSession.MediaKind.IMAGE
        assert session.source_ref == "face.jpg"

    def test_audio_upload_accepted_with_kind(self, tmp_path, settings) -> None:
        settings.MEDIA_ROOT = str(tmp_path)
        upload = SimpleUploadedFile(
            "speech.wav", b"fake-wav-bytes", content_type="audio/x-wav"
        )
        result = UploadService(probe=lambda _p: True).submit_file(upload)

        assert result["accepted"] is True
        assert result["media_kind"] == "audio"
        session = AnalysisSession.objects.get(id=result["session_id"])
        assert session.media_kind == AnalysisSession.MediaKind.AUDIO

    def test_video_upload_still_defaults_to_video_kind(self, tmp_path, settings) -> None:
        settings.MEDIA_ROOT = str(tmp_path)
        upload = SimpleUploadedFile("clip.mp4", b"bytes", content_type="video/mp4")
        result = UploadService(probe=lambda _p: True).submit_file(upload)

        assert result["accepted"] is True
        assert result["media_kind"] == "video"

    def test_real_probe_rejects_invalid_image(self, tmp_path) -> None:
        """A `.jpg` whose bytes do not decode as an image is UNDECODABLE."""
        from detection.services.upload import _image_probe

        upload = SimpleUploadedFile("broken.jpg", b"not-an-image")
        result = UploadService(probe=_image_probe).submit_file(upload)

        assert result["accepted"] is False
        assert result["error_code"] == "UNDECODABLE"
        assert "image" in (result["message"] or "").lower()
        assert AnalysisSession.objects.count() == 0


# ---------------------------------------------------------------------------
# URL submission is gone
# ---------------------------------------------------------------------------


class TestUrlRemoved:
    def test_url_payload_rejected_with_typed_error(self) -> None:
        response = APIClient().post(
            "/api/analyses",
            {"url": "https://example.com/video.mp4"},
            format="json",
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.data["error_code"] == "URL_NOT_SUPPORTED"
        assert AnalysisSession.objects.count() == 0

    def test_no_input_rejected(self) -> None:
        response = APIClient().post("/api/analyses", {}, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.data["error_code"] == "NO_INPUT"


# ---------------------------------------------------------------------------
# GET /api/analyses/{id}: last-analysis metrics endpoint
# ---------------------------------------------------------------------------


class TestGetAnalysisEndpoint:
    def _make_session(self) -> AnalysisSession:
        from detection.models import AudioResult, FusionResult, VisualResult

        session = AnalysisSession.objects.create(
            source_type=AnalysisSession.SourceType.FILE,
            source_ref="face.jpg",
            media_kind=AnalysisSession.MediaKind.IMAGE,
            status=AnalysisSession.Status.COMPLETED,
        )
        VisualResult.objects.create(
            session=session,
            aggregate_likelihood=0.72,
            state=VisualResult.State.OK,
            frames_analyzed=1,
            faces_isolated=1,
        )
        AudioResult.objects.create(
            session=session,
            likelihood=None,
            state=AudioResult.State.NO_AUDIO_SIGNAL,
        )
        FusionResult.objects.create(
            session=session,
            score=0.72,
            label=FusionResult.Label.DEEPFAKE,
            modalities_used="visual",
            inconclusive=False,
            threshold_used=0.5,
        )
        return session

    def test_returns_last_analysis_metrics(self) -> None:
        session = self._make_session()
        response = APIClient().get(f"/api/analyses/{session.id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.data["media_kind"] == "image"
        assert response.data["visual"]["likelihood"] == pytest.approx(0.72)
        assert response.data["visual"]["frames_analyzed"] == 1
        assert response.data["audio"]["state"] == "NO_AUDIO_SIGNAL"
        assert response.data["fusion"]["score"] == pytest.approx(0.72)
        assert response.data["fusion"]["modalities_used"] == ["visual"]

    def test_unknown_session_returns_404(self) -> None:
        import uuid

        response = APIClient().get(f"/api/analyses/{uuid.uuid4()}")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_health_no_longer_advertises_url_flag(self) -> None:
        response = APIClient().get("/api/health")
        assert response.status_code == status.HTTP_200_OK
        assert "external_url_enabled" not in response.data["feature_flags"]
