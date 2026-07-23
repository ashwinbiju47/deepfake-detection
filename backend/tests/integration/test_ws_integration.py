"""Integration tests for WebSocket connection and timing (Task 18.1, Requirements 5.1, 5.2, 5.3, 5.4, 7.1, 7.2, 7.3, 11.3).
"""

from __future__ import annotations

import pytest
from channels.testing import WebsocketCommunicator

from config.asgi import application
from detection.models import AnalysisSession
from detection.services.streamer import StreamService

pytestmark = [pytest.mark.django_db, pytest.mark.asyncio]


async def test_websocket_communicator_connection_and_event_stream() -> None:
    """Requirement 5.1, 5.3: Client connects to /ws/analyses/{id}/ and receives stream events."""
    session = await AnalysisSession.objects.acreate(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.PROCESSING,
    )

    communicator = WebsocketCommunicator(
        application, f"/ws/analyses/{session.id}/"
    )
    connected, _subprotocol = await communicator.connect()
    assert connected is True

    # Publish progress event
    await StreamService.publish_event(
        str(session.id), "progress", {"progress_percent": 25.0}
    )

    response = await communicator.receive_json_from(timeout=2.0)
    assert response["type"] == "progress"
    assert response["progress_percent"] == 25.0

    await communicator.disconnect()
