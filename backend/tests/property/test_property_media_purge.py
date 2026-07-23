"""Property 15: Media purge removes all media artifacts across all triggers (Task 11.2, Requirements 9.1, 9.2, 9.4).

Feature: deepfake-detection-platform, Property 15: Media purge removes all media artifacts across all triggers
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.models import AnalysisSession, PurgeRecord
from detection.services.purger import MediaPurger
from detection.storage import session_media_dir

pytestmark = [pytest.mark.django_db, pytest.mark.property]


@given(dummy_filename=st.sampled_from(["file1.mp4", "file2.avi", "chunk.dat"]))
@settings(max_examples=100)
def test_property_media_purge_removes_artifacts(dummy_filename: str) -> None:
    """Property 15: Purging a session deletes all transient media and marks PURGED."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="input.mp4",
        status=AnalysisSession.Status.COMPLETED,
        media_state=AnalysisSession.MediaState.PRESENT,
    )

    media_dir = session_media_dir(session.id)
    media_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = media_dir / dummy_filename
    dummy_file.write_bytes(b"dummy_media_bytes")

    record = MediaPurger.purge(str(session.id))

    assert record.status == PurgeRecord.Status.SUCCESS
    assert record.verified_empty is True
    assert not media_dir.exists()

    session.refresh_from_db()
    assert session.media_state == AnalysisSession.MediaState.PURGED
