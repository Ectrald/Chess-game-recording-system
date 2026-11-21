"""WebSocket Connection Manager for Chess Game Recording System Web UI.

Manages active browser connections and broadcasts real-time state updates,
move detections, UNCERTAIN arbitrations, and telemetry metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Thread-safe WebSocket broadcaster managing live operator connections."""

    def __init__(self) -> None:
        """Initializes empty connection pool and event loop reference."""
        self._active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Registers the running asyncio event loop for threadsafe dispatch."""
        self._loop = loop

    @property
    def active_count(self) -> int:
        """Returns the number of currently connected WebSocket clients."""
        return len(self._active_connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and registers a new client WebSocket connection."""
        await websocket.accept()
        self._active_connections.add(websocket)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        logger.info("WebSocket client connected. Total clients: %d", len(self._active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Removes a client WebSocket from the active connection pool."""
        self._active_connections.discard(websocket)
        logger.info("WebSocket client disconnected. Total clients: %d", len(self._active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcasts a JSON-serializable message payload to all active clients."""
        if not self._active_connections:
            return

        payload = json.dumps(message)
        dead_sockets: List[WebSocket] = []

        for ws in list(self._active_connections):
            try:
                await ws.send_text(payload)
            except Exception as exc:
                logger.debug("Failed to send WebSocket message to client (%s), pruning.", exc)
                dead_sockets.append(ws)

        for ws in dead_sockets:
            self._active_connections.discard(ws)

    def broadcast_threadsafe(self, message: Dict[str, Any]) -> None:
        """Dispatches broadcast from background OS thread into the asyncio event loop."""
        if self._loop is not None and self._loop.is_running() and self._active_connections:
            try:
                asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)
            except RuntimeError as err:
                logger.debug("Could not schedule WebSocket broadcast: %s", err)
