"""Result_Streamer: WebSocket event logging and publishing service (Task 10.1, Requirements 5, 7).

Publishes progress, heatmap, result, and error events to the session group
channel layer and records sequence-numbered StreamEvent log instances.
"""

from __future__ import annotations

from typing import Any, Dict
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db import transaction

from detection.models import AnalysisSession, StreamEvent


class StreamService:
    """Service to publish stream events to WebSocket clients and record StreamEvent log."""

    @staticmethod
    def publish_event(
        session_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> StreamEvent | None:
        """Record StreamEvent with auto-increment sequence and broadcast over channel layer."""
        try:
            with transaction.atomic():
                session = AnalysisSession.objects.get(id=session_id)
                last_event = (
                    StreamEvent.objects.filter(session=session)
                    .order_by("-seq")
                    .first()
                )
                seq = (last_event.seq + 1) if last_event else 1

                full_payload = dict(payload)
                full_payload["seq"] = seq
                full_payload["type"] = event_type
                full_payload["session_id"] = str(session_id)

                event = StreamEvent.objects.create(
                    session=session,
                    seq=seq,
                    type=event_type,
                    payload=full_payload,
                )

            # Broadcast via Channels group
            channel_layer = get_channel_layer()
            if channel_layer:
                group_name = f"analysis_{session_id}"
                async_to_sync(channel_layer.group_send)(
                    group_name,
                    {
                        "type": "stream.event",
                        "payload": full_payload,
                    },
                )

            return event
        except Exception:
            return None

    @staticmethod
    def get_events_after(session_id: str, last_seq: int) -> list[dict[str, Any]]:
        """Retrieve recorded stream events for session_id with seq > last_seq (Requirement 5.5)."""
        events = StreamEvent.objects.filter(
            session_id=session_id, seq__gt=last_seq
        ).order_by("seq")
        return [e.payload for e in events]
