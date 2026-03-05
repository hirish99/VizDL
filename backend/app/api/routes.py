"""REST API routes."""
import asyncio
import json
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..config import settings
from ..engine.executor import execute_graph
from ..engine.graph import Edge, Graph, NodeInstance
from ..engine.pipeline import execute_pipeline
from ..engine.validator import validate_graph
from ..models.schemas import (
    AutoBatchSizeRequest, AutoBatchSizeResponse,
    ExecuteRequest, ExecuteResponse, GraphSchema,
    SavedGraph, UploadResponse,
)
from ..engine.session import create_session, get_session, remove_session
from ..nodes.registry import NodeRegistry
from .websocket import manager

router = APIRouter(prefix="/api")
executor_pool = ThreadPoolExecutor(max_workers=4)

# In-memory stores (capped at _MAX_RESULTS to prevent unbounded growth)
_MAX_RESULTS = 20
_results: dict[str, Any] = {}


def _schema_to_graph(schema: GraphSchema) -> Graph:
    nodes = {
        n.id: NodeInstance(
            id=n.id, node_type=n.node_type,
            params=n.params, disabled=n.disabled,
            position=n.position,
        )
        for n in schema.nodes
    }
    edges = [
        Edge(
            id=e.id, source_node=e.source_node, source_output=e.source_output,
            target_node=e.target_node, target_input=e.target_input, order=e.order,
        )
        for e in schema.edges
    ]
    return Graph(nodes=nodes, edges=edges)


@router.get("/nodes")
async def list_nodes():
    """Return all registered node definitions."""
    defs = NodeRegistry.all_definitions()
    result = {}
    for name, defn in defs.items():
        result[name] = {
            "node_type": defn.node_type,
            "display_name": defn.display_name,
            "category": defn.category,
            "description": defn.description,
            "inputs": {
                k: {
                    "dtype": v.dtype.value,
                    "default": v.default,
                    "required": v.required,
                    "min_val": v.min_val,
                    "max_val": v.max_val,
                    "choices": v.choices,
                    "is_handle": v.is_handle,
                }
                for k, v in defn.inputs.items()
            },
            "outputs": [
                {"dtype": o.dtype.value, "name": o.name}
                for o in defn.outputs
            ],
        }
    return result


@router.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    """Execute the training pipeline: layer graph + config.

    Returns immediately with execution_id.  Training runs in background;
    progress, completion, and errors are delivered via WebSocket.
    """
    layer_graph = _schema_to_graph(request.graph)
    config = request.config.model_dump()

    session_id = request.session_id or str(uuid.uuid4())
    execution_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    _raw_cb = manager.make_progress_callback(session_id, loop)

    def progress_cb(data: dict):
        data["execution_id"] = execution_id
        _raw_cb(data)

    session = create_session(execution_id, session_id)
    checkpoint_path = settings.weights_dir / "checkpoints" / f"{execution_id}.pt"

    def run():
        return execute_pipeline(
            layer_graph, config,
            progress_callback=progress_cb,
            training_controller=session.controller,
            checkpoint_path=checkpoint_path,
        )

    async def _run_training():
        try:
            await manager.send_to_session(session_id, {
                "type": "execution_start", "execution_id": execution_id,
            })

            results = await loop.run_in_executor(executor_pool, run)

            serialized = _serialize_pipeline_results(results)
            del results  # Drop reference so model/tensors can be GC'd
            # Evict oldest entries if at capacity
            while len(_results) >= _MAX_RESULTS:
                _results.pop(next(iter(_results)))
            _results[execution_id] = serialized

            await manager.send_to_session(session_id, {
                "type": "execution_complete",
                "execution_id": execution_id,
                "results": serialized,
            })
        except Exception as e:
            await manager.send_to_session(session_id, {
                "type": "execution_error",
                "execution_id": execution_id,
                "error": str(e),
            })
        finally:
            remove_session(execution_id)

    asyncio.create_task(_run_training())

    return ExecuteResponse(
        execution_id=execution_id, status="started", results={},
    )


def _serialize_pipeline_results(results: dict[str, Any]) -> dict[str, Any]:
    """Convert pipeline results to JSON-serializable format."""
    serialized = {}
    for key, value in results.items():
        if isinstance(value, dict):
            clean = {}
            for k, v in value.items():
                if isinstance(v, (str, int, float, bool, list, type(None))):
                    clean[k] = v
                elif isinstance(v, dict):
                    clean[k] = v
            serialized[key] = clean
        elif isinstance(value, (str, int, float, bool, list, type(None))):
            serialized[key] = value
        else:
            serialized[key] = str(type(value).__name__)
    return serialized


@router.post("/execute/{execution_id}/pause")
async def pause_training(execution_id: str):
    session = get_session(execution_id)
    if not session:
        raise HTTPException(status_code=404, detail="Execution not found or already completed")
    session.controller.pause()
    await manager.send_to_session(session.session_id, {
        "type": "training_paused", "execution_id": execution_id,
    })
    return {"status": "paused"}


@router.post("/execute/{execution_id}/resume")
async def resume_training(execution_id: str):
    session = get_session(execution_id)
    if not session:
        raise HTTPException(status_code=404, detail="Execution not found or already completed")
    session.controller.resume()
    await manager.send_to_session(session.session_id, {
        "type": "training_resumed", "execution_id": execution_id,
    })
    return {"status": "resumed"}


@router.post("/execute/{execution_id}/stop")
async def stop_training(execution_id: str):
    session = get_session(execution_id)
    if not session:
        raise HTTPException(status_code=404, detail="Execution not found or already completed")
    session.controller.stop()
    await manager.send_to_session(session.session_id, {
        "type": "training_stopped", "execution_id": execution_id,
    })
    return {"status": "stopped"}


@router.get("/results/{execution_id}")
async def get_results(execution_id: str):
    if execution_id not in _results:
        raise HTTPException(status_code=404, detail="Execution not found")
    return _results[execution_id]


@router.post("/auto-batch-size", response_model=AutoBatchSizeResponse)
async def auto_batch_size(request: AutoBatchSizeRequest):
    """Find max batch size via GPU binary search (forward+backward).

    Runs in a subprocess so all GPU memory is freed when the process exits,
    preventing OOM leaks from corrupted autograd graphs.
    """
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="No CUDA GPU available")

    # Resolve input_dim from file if file_id + input_columns are provided
    input_dim = request.input_dim
    if request.file_id and request.input_columns:
        from ..nodes.data import _resolve_columns
        file_path = settings.upload_dir / request.file_id
        if not file_path.exists():
            raise HTTPException(status_code=400, detail=f"Data file not found: {request.file_id}")
        if file_path.suffix == ".pt":
            loaded = torch.load(file_path, map_location="cpu", weights_only=False)
            available = loaded["header"]
            del loaded
        else:
            import pandas as pd
            available = list(pd.read_csv(file_path, nrows=0).columns)
        specs = [s.strip() for s in request.input_columns.split(",") if s.strip()]
        try:
            resolved = _resolve_columns(specs, available)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        input_dim = len(resolved)

    if input_dim is None or input_dim < 1:
        raise HTTPException(status_code=400, detail="Could not determine input dimension — provide file_id + input_columns or input_dim")

    max_cap = None
    if request.num_train_samples and request.num_train_samples > 0:
        max_cap = request.num_train_samples

    # Serialize config for subprocess
    worker_config = {
        "graph": request.graph.model_dump(),
        "input_dim": input_dim,
        "optimizer": request.optimizer,
        "max_cap": max_cap,
    }
    config_json = json.dumps(worker_config)

    def _run_in_subprocess():
        proc = subprocess.run(
            [sys.executable, "-m", "backend.app.api._auto_batch_worker"],
            input=config_json,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).resolve().parents[3]),  # project root
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            return {"error": f"Auto batch size search failed: {stderr or 'unknown error'}"}
        return json.loads(proc.stdout)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor_pool, _run_in_subprocess)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    max_bs = result["max_bs"]
    search_log = result["log"]

    if max_bs <= 1:
        raise HTTPException(status_code=400, detail="Model too large for even batch_size=2 on this GPU")

    return AutoBatchSizeResponse(
        max_batch_size=max_bs,
        steps_tried=len(search_log),
        search_log=search_log,
    )


@router.get("/upload/server-files")
async def list_server_files():
    """List data files already on the server's upload directory."""
    files = []
    for ext in ("*.csv", "*.pt"):
        for p in sorted(settings.upload_dir.glob(ext)):
            files.append({"file_id": p.name, "filename": p.name, "size_mb": round(p.stat().st_size / (1024 * 1024), 1)})
    return files


@router.post("/upload/server-files/{file_id}")
async def use_server_file(file_id: str):
    """Use a data file already on the server — returns columns and row count like upload does."""
    dest = settings.upload_dir / file_id
    if not dest.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if dest.suffix == ".pt":
        loaded = torch.load(dest, map_location="cpu", weights_only=False)
        columns = loaded["header"]
        rows = loaded["data"].shape[0]
        del loaded
    else:
        columns = list(pd.read_csv(dest, nrows=0).columns)
        rows = sum(len(chunk) for chunk in pd.read_csv(dest, usecols=[0], chunksize=100_000))

    return UploadResponse(
        file_id=file_id,
        filename=file_id,
        columns=columns,
        rows=rows,
    )


@router.post("/upload/csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV or .pt file, return file_id and column info."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".pt"):
        raise HTTPException(status_code=400, detail="Only .csv and .pt files are supported")

    file_id = f"{uuid.uuid4()}_{file.filename}"
    dest = settings.upload_dir / file_id
    # Stream to disk to avoid holding entire file in memory
    with open(dest, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):  # 8MB chunks
            f.write(chunk)

    if ext == ".pt":
        loaded = torch.load(dest, map_location="cpu", weights_only=False)
        columns = loaded["header"]
        rows = loaded["data"].shape[0]
        del loaded
    else:
        columns = list(pd.read_csv(dest, nrows=0).columns)
        rows = sum(len(chunk) for chunk in pd.read_csv(dest, usecols=[0], chunksize=100_000))

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        columns=columns,
        rows=rows,
    )


@router.post("/graphs")
async def save_graph(graph: SavedGraph):
    """Save a named graph configuration to disk."""
    if not graph.id:
        graph.id = str(uuid.uuid4())
    path = settings.graphs_dir / f"{graph.id}.json"
    path.write_text(graph.model_dump_json(indent=2))
    return {"id": graph.id}


@router.get("/graphs")
async def list_graphs():
    result = {}
    for path in settings.graphs_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            gid = data["id"]
            result[gid] = {"id": gid, "name": data.get("name", ""), "description": data.get("description", "")}
        except Exception:
            continue
    return result


@router.get("/graphs/{graph_id}")
async def get_graph(graph_id: str):
    path = settings.graphs_dir / f"{graph_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Graph not found")
    return json.loads(path.read_text())


@router.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str):
    path = settings.graphs_dir / f"{graph_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Graph not found")
    path.unlink()
    return {"status": "deleted"}


@router.get("/models")
async def list_models():
    """List saved models available for continue-training."""
    models = []
    for report_path in sorted(settings.weights_dir.glob("*_report.json"), reverse=True):
        weights_path = report_path.with_name(report_path.name.replace("_report.json", ".pt"))
        if not weights_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text())
            training = report.get("training", {})
            models.append({
                "path": str(weights_path),
                "name": weights_path.stem,
                "architecture": report.get("model", {}).get("architecture", "unknown"),
                "parameter_count": report.get("model", {}).get("parameter_count"),
                "final_train_loss": training.get("final_train_loss"),
                "final_val_loss": training.get("final_val_loss"),
                "total_epochs": training.get("total_epochs", len(training.get("epochs", training.get("epoch", [])))),
                "timestamp": report.get("timestamp", ""),
                "graph": report.get("graph"),
                "config": report.get("pipeline_config"),
            })
        except Exception:
            continue
    return models
