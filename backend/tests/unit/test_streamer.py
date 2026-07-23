"""Unit tests for WebSocket event streamer and replay (Task 10.1, 10.3, Requirements 5.3, 5.4, 5.5, 7.4).
"""

from __future__ import annotations

import pytest

from detection.models import AnalysisSession, StreamEvent
from detection.services.streamer import StreamService

pytestmark = [pytest.mark.django_db, pytest.mark.unit]


def test_stream_service_publishes_and_logs_events() -> None:
    """Requirement 5.3, 5.4: Events are saved with sequential sequence numbers."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.PROCESSING,
    )

    ev1 = StreamService.publish_event(str(session.id), "progress", {"progress_percent": 0.0})
    ev2 = StreamService.publish_event(str(session.id), "progress", {"progress_percent": 50.0})

    assert ev1 is not None and ev2 is not None
    assert ev1.seq == 1
    assert ev2.seq == 2

    events = list(StreamEvent.objects.filter(session=session).order_by("seq"))
    assert len(events) == 2
    assert events[0].type == "progress"
    assert events[1].payload["progress_percent"] == 50.0


def test_stream_service_replay_resume() -> None:
    """Requirement 5.5: get_events_after returns events with seq > last_seq."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.PROCESSING,
    )

    StreamService.publish_event(str(session.id), "progress", {"progress_percent": 0.0})
    StreamService.publish_event(str(session.id), "progress", {"progress_percent": 50.0})
    StreamService.publish_event(str(session.id), "result", {"score": 0.8, "label": "deepfake"})

    replayed = StreamService.get_events_after(str(session.id), last_seq=1)
    assert len(replayed) == 2
    assert replayed[0]["seq"] == 2
    assert replayed[1]["seq"] == 3
    assert replayed[1]["type"] == "result"
