"""FastAPI application with CORS, lifespan, and routes."""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.routes import router
from .api.websocket import manager
from .api.system_monitor import system_monitor_ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: trigger node auto-discovery
    from . import nodes  # noqa: F401
    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.websocket("/ws/training/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            # Keep connection alive, handle any client messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)


@app.websocket("/ws/system/{session_id}")
async def system_monitor_endpoint(websocket: WebSocket, session_id: str):
    await system_monitor_ws(websocket, session_id)
