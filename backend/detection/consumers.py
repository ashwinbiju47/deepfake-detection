"""WebSocket consumers (Result_Streamer) (Task 10.1, Requirements 5, 7).

Streams progress, result, and error events to connected clients and supports
reconnection replay via `resume {last_seq}` messages.
"""

from __future__ import annotations

from typing import Any, Dict
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from detection.services.streamer import StreamService


class AnalysisConsumer(AsyncJsonWebsocketConsumer):
    """Streams progress/results for a single ``Analysis_Session``."""

    async def connect(self) -> None:
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self.group_name = f"analysis_{self.session_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, code: int) -> None:
        group_name = getattr(self, "group_name", None)
        if group_name is not None:
            await self.channel_layer.group_discard(group_name, self.channel_name)

    async def receive_json(self, content: Dict[str, Any], **kwargs: Any) -> None:
        """Handle incoming messages from WebSocket client (e.g. resume command)."""
        action = content.get("action")
        if action == "resume":
            last_seq = content.get("last_seq", 0)
            missed_events = await database_sync_to_async(StreamService.get_events_after)(
                self.session_id, last_seq
            )
            for event_payload in missed_events:
                await self.send_json(event_payload)

    async def stream_event(self, event: Dict[str, Any]) -> None:
        """Relay an event published to the session group to the client."""
        await self.send_json(event.get("payload", {}))
