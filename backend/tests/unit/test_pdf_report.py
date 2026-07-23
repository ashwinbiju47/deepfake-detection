"""Unit tests for PDF ReportGenerator (Task 16.3, Requirements 12.1, 12.2, 12.5).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from detection.models import AnalysisSession, FusionResult, Report
from detection.services.report import ReportGenerator

pytestmark = [pytest.mark.django_db, pytest.mark.unit]


def test_pdf_report_disabled_rejection() -> None:
    """Requirement 12.1: Report generation rejected when REPORT_ENABLED is False."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.COMPLETED,
    )
    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.REPORT_ENABLED", False)
        res = ReportGenerator.generate(str(session.id))

        assert res.success is False
        assert res.error_code == "REPORT_DISABLED"


def test_pdf_report_generation_success() -> None:
    """Requirement 12.1, 12.2: Completed session generates PDF bytes and marks Report AVAILABLE."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.COMPLETED,
    )
    FusionResult.objects.create(
        session=session,
        score=0.85,
        label="deepfake",
        modalities_used="visual,audio",
        threshold_used=0.5,
    )

    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.REPORT_ENABLED", True)
        res = ReportGenerator.generate(str(session.id))

        assert res.success is True
        assert res.pdf_bytes is not None

        rep = Report.objects.get(session=session)
        assert rep.status == Report.Status.AVAILABLE


def test_pdf_report_failure_delivers_no_partial_pdf() -> None:
    """Requirement 12.5: Report generation failure produces no partial PDF."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.COMPLETED,
    )

    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.REPORT_ENABLED", True)
        with patch("detection.services.report.ReportGenerator._build_pdf", side_effect=RuntimeError("ReportLab crashed")):
            res = ReportGenerator.generate(str(session.id))

            assert res.success is False
            assert res.pdf_bytes is None
            assert res.error_code == "REPORT_GENERATION_FAILED"

            rep = Report.objects.get(session=session)
            assert rep.status == Report.Status.FAILED
