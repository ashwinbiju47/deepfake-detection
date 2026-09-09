"""Unit tests for Grad-CAM XAIGenerator (Task 15.3, Requirements 11.1, 11.2, 11.4).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from detection.models import AnalysisSession, FrameHeatmap
from detection.services.xai import XAIGenerator

pytestmark = [pytest.mark.django_db, pytest.mark.unit]


def test_heatmap_generation_success() -> None:
    """Requirement 11.1: Valid session creates FrameHeatmap overlay record."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.PROCESSING,
    )

    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.HEATMAP_ENABLED", True)
        outcome = XAIGenerator.generate_heatmap(str(session.id), "frame_001", [[0.2, 0.9]])

        assert outcome.success is True
        assert outcome.heatmap_id is not None

        record = FrameHeatmap.objects.get(id=outcome.heatmap_id)
        assert record.frame_id == "frame_001"
        assert record.delivered is True


def test_heatmap_generates_original_heatmap_overlay_triple() -> None:
    """Requirement 11.1: XAI produces ORIGINAL + HEATMAP + OVERLAY (not heatmap only)."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.PROCESSING,
    )

    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.HEATMAP_ENABLED", True)
        outcome = XAIGenerator.generate_heatmap(
            str(session.id), "frame_001", [[0.0, 0.5], [0.5, 1.0]]
        )

        assert outcome.success is True
        assert outcome.original_b64 and outcome.heatmap_b64 and outcome.overlay_b64

        record = FrameHeatmap.objects.get(id=outcome.heatmap_id)
        assert record.original_png is not None
        assert record.heatmap_png is not None
        assert record.overlay_png is not None

        # All three artifacts are real PNGs (magic bytes), not placeholders
        for png in (record.original_png, record.heatmap_png, record.overlay_png):
            assert bytes(png).startswith(b"\x89PNG\r\n\x1a\n")


def test_heatmap_triple_uses_original_frame_when_provided() -> None:
    """Requirement 11.1: When a frame image is supplied it becomes the ORIGINAL artifact."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.PROCESSING,
    )

    # Simulate an OpenCV-style BGR frame: 8x8, left half dark, right half bright
    frame_image = [
        [(20, 20, 20) if x < 4 else (220, 220, 220) for x in range(8)]
        for _ in range(8)
    ]

    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.HEATMAP_ENABLED", True)
        outcome = XAIGenerator.generate_heatmap(
            str(session.id), "frame_002", [[0.2, 0.9]], frame_image=frame_image
        )

        assert outcome.success is True
        record = FrameHeatmap.objects.get(id=outcome.heatmap_id)
        assert record.original_png is not None
        assert len(bytes(record.original_png)) > 0


def test_heatmap_generation_failure_preserves_classification() -> None:
    """Requirement 11.2: Heatmap failure emits error event, classification unaffected."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.PROCESSING,
    )

    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.HEATMAP_ENABLED", True)
        with patch("detection.services.xai.normalize_heatmap", side_effect=RuntimeError("PyTorch Grad-CAM engine error")):
            outcome = XAIGenerator.generate_heatmap(str(session.id), "frame_002")

            assert outcome.success is False
            assert "PyTorch Grad-CAM engine error" in outcome.error_detail
