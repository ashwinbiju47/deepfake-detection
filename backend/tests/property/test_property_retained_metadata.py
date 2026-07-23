"""Property 16: Only non-media metadata is retained after purge (Task 11.3, Requirement 9.3).

Feature: deepfake-detection-platform, Property 16: Only non-media metadata is retained after purge
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.models import AnalysisSession, FusionResult
from detection.services.purger import MediaPurger
from detection.storage import session_media_dir

pytestmark = [pytest.mark.django_db, pytest.mark.property]


@given(
    score=st.floats(min_value=0.0, max_value=1.0),
    label=st.sampled_from(["authentic", "deepfake"]),
)
@settings(max_examples=100)
def test_property_retained_metadata_after_purge(score: float, label: str) -> None:
    """Property 16: Non-media metadata (score, label, timestamps) is preserved while media is deleted."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.COMPLETED,
        media_state=AnalysisSession.MediaState.PRESENT,
    )
    FusionResult.objects.create(
        session=session,
        score=score,
        label=label,
        modalities_used="visual,audio",
        threshold_used=0.5,
    )

    media_dir = session_media_dir(session.id)
    media_dir.mkdir(parents=True, exist_ok=True)
    (media_dir / "frame.jpg").write_bytes(b"frame_bytes")

    MediaPurger.purge(str(session.id))

    # Media directory is gone
    assert not media_dir.exists()

    # DB Metadata is intact
    session.refresh_from_db()
    assert session.source_ref == "video.mp4"
    assert session.fusion_result.score == pytest.approx(score)
    assert session.fusion_result.label == label
