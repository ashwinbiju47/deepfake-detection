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
