"""Channels WebSocket routing for the detection app."""

from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(
        r"^ws/analyses/(?P<session_id>[0-9a-zA-Z\-]+)/$",
        consumers.AnalysisConsumer.as_asgi(),
    ),
]
