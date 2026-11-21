"""Web UI package for Chess Game Recording System.

Provides FastAPI-based HTTP REST, MJPEG streaming, WebSocket live updates,
and responsive browser-based operator dashboard.
"""

from __future__ import annotations

from src.ui.web.app import create_web_app
from src.ui.web.streamer import MJPEGStreamer
from src.ui.web.ws_manager import WebSocketManager

__all__ = [
    "create_web_app",
    "MJPEGStreamer",
    "WebSocketManager",
]
