"""WebSocket connection manager for real-time training telemetry."""
import asyncio
import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Manages active WebSocket connections per session."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = []
        self._connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self._connections:
            self._connections[session_id] = [
                ws for ws in self._connections[session_id] if ws is not websocket
            ]
            if not self._connections[session_id]:
                del self._connections[session_id]

    async def send_to_session(self, session_id: str, data: dict[str, Any]):
        if session_id not in self._connections:
            return
        message = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in self._connections[session_id]:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(session_id, ws)

    def make_progress_callback(self, session_id: str, loop: asyncio.AbstractEventLoop):
        """Create a sync callback that sends progress via WebSocket from a thread."""
        def callback(data: dict[str, Any]):
            asyncio.run_coroutine_threadsafe(
                self.send_to_session(session_id, data),
                loop,
            )
        return callback


manager = ConnectionManager()
