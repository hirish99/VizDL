"""WebSocket endpoint that streams CPU/RAM/GPU stats every 2 seconds."""
import asyncio
import json

import psutil
from fastapi import WebSocket, WebSocketDisconnect

# Try to init NVML once at import time
_nvml_handle = None
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pass


async def system_monitor_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            stats = _collect_stats()
            await websocket.send_text(json.dumps(stats))
            await asyncio.sleep(2)
    except (WebSocketDisconnect, Exception):
        pass


def _collect_stats() -> dict:
    mem = psutil.virtual_memory()
    data: dict = {
        "type": "system_status",
        "cpu": psutil.cpu_percent(interval=None),
        "ram": mem.percent,
        "gpu_util": None,
        "gpu_mem_used": None,
        "gpu_mem_total": None,
    }
    if _nvml_handle is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            data["gpu_util"] = util.gpu
            data["gpu_mem_used"] = round(mem_info.used / (1024 ** 3), 1)
            data["gpu_mem_total"] = round(mem_info.total / (1024 ** 3), 1)
        except Exception:
            pass
    return data
