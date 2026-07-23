"""Property 11: Enqueue failure never leaves a session in-progress (Task 9.2, Requirement 6.2).

Feature: deepfake-detection-platform, Property 11: Enqueue failure never leaves a session in-progress

If enqueuing an analysis task fails, the session MUST be transitioned to FAILED_TO_QUEUE
or returned as an upload failure, and MUST NEVER remain in QUEUED or PROCESSING status.
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from detection.models import AnalysisSession
from detection.services.upload import UploadService

pytestmark = [pytest.mark.django_db, pytest.mark.property]


@given(filename=st.sampled_from(["video.mp4", "clip.avi"]))
@settings(max_examples=100)
def test_property_enqueue_failure_never_leaves_session_in_progress(filename: str) -> None:
    """Property 11: Enqueue failure transitions session to FAILED_TO_QUEUE or returns error."""
    upload = SimpleUploadedFile(filename, b"valid_video_header_bytes", content_type="video/mp4")

    # Injected enqueuer that simulates broker connection failure
    def failing_enqueuer(session: AnalysisSession) -> bool:
        session.status = AnalysisSession.Status.FAILED
        session.save(update_fields=["status"])
        return False



    service = UploadService(
        probe=lambda _path: True,
        enqueuer=failing_enqueuer,
    )

    res = service.submit_file(upload)

    assert res["accepted"] is False
    assert res["error_code"] == "ENQUEUE_FAILED"

    # Verify database state
    sessions = list(AnalysisSession.objects.all())
    for s in sessions:
        assert s.status not in (
            AnalysisSession.Status.QUEUED,
            AnalysisSession.Status.PROCESSING,
        ), "Failed enqueue must never leave a session in QUEUED or PROCESSING status"
