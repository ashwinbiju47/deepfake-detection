"""Property 22: Reports for completed sessions contain required content; incomplete sessions are refused (Task 16.2, Requirements 12.1, 12.3, 12.4).

Feature: deepfake-detection-platform, Property 22: Reports for completed sessions contain required content; incomplete sessions are refused
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.models import AnalysisSession, FusionResult
from detection.services.report import ReportGenerator

pytestmark = [pytest.mark.django_db, pytest.mark.property]


@given(status_val=st.sampled_from(["QUEUED", "PROCESSING", "COMPLETED", "INCONCLUSIVE"]))
@settings(max_examples=100)
def test_property_pdf_report_session_status_policy(status_val: str) -> None:
    """Property 22: Incomplete sessions (QUEUED/PROCESSING) are refused; completed sessions return PDF."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=status_val,
    )
    if status_val in ("COMPLETED", "INCONCLUSIVE"):
        FusionResult.objects.create(
            session=session,
            score=0.75,
            label="deepfake",
            modalities_used="visual,audio",
            threshold_used=0.5,
        )

    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.REPORT_ENABLED", True)
        res = ReportGenerator.generate(str(session.id))

        if status_val in ("QUEUED", "PROCESSING"):
            assert res.success is False
            assert res.error_code == "SESSION_INCOMPLETE"
            assert res.pdf_bytes is None
        else:
            assert res.success is True
            assert res.pdf_bytes is not None
            assert len(res.pdf_bytes) > 0
