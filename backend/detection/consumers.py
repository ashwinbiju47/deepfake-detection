"""WebSocket consumers (Result_Streamer).

A minimal consumer is provided so the ASGI routing is valid in the skeleton.
The full event-log / replay logic (Requirements 5.x, 7.x) is implemented in
Task 10.1.
"""

from __future__ import annotations

from channels.generic.websocket import AsyncJsonWebsocketConsumer


class AnalysisConsumer(AsyncJsonWebsocketConsumer):
    """Streams progress/results for a single ``Analysis_Session``.

    Each session uses the channel-layer group ``analysis_{session_id}`` so
    worker-published events fan out to the connected dashboard.
    """

    async def connect(self) -> None:
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self.group_name = f"analysis_{self.session_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, code: int) -> None:
        group_name = getattr(self, "group_name", None)
        if group_name is not None:
            await self.channel_layer.group_discard(group_name, self.channel_name)

    async def stream_event(self, event: dict) -> None:
        """Relay an event published to the session group to the client."""
        await self.send_json(event.get("payload", {}))
