"""Pydantic schemas for API request/response models."""
from typing import Any
from pydantic import BaseModel


class NodeParamSchema(BaseModel):
    key: str
    value: Any


class EdgeSchema(BaseModel):
    id: str
    source_node: str
    source_output: int = 0
    target_node: str
    target_input: str
    order: int = 0


class NodeSchema(BaseModel):
    id: str
    node_type: str
    params: dict[str, Any] = {}
    disabled: bool = False
    position: dict[str, float] = {}


class GraphSchema(BaseModel):
    nodes: list[NodeSchema]
    edges: list[EdgeSchema]
    name: str = ""
    description: str = ""


class PipelineConfig(BaseModel):
    # Data
    file_id: str
    input_columns: str
    target_columns: str
    val_ratio: float = 0.2
    batch_size: int = 32
    shuffle: bool = True
    # Training
    loss_fn: str = "MSELoss"
    optimizer: str = "Adam"
    lr: float = 0.001
    epochs: int = 10
    # Export
    export_name: str = ""
    # Test data (optional)
    test_file_id: str | None = None
    test_input_columns: str | None = None
    test_target_columns: str | None = None


class ExecuteRequest(BaseModel):
    graph: GraphSchema
    config: PipelineConfig
    session_id: str | None = None


class ExecuteResponse(BaseModel):
    execution_id: str
    status: str
    results: dict[str, Any] = {}
    errors: list[str] = []


class NodeDefinitionResponse(BaseModel):
    node_type: str
    display_name: str
    category: str
    description: str
    inputs: dict[str, Any]
    outputs: list[dict[str, Any]]


class SavedGraph(BaseModel):
    id: str
    name: str
    description: str = ""
    graph: GraphSchema
    config: PipelineConfig | None = None


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    columns: list[str]
    rows: int


class VramEstimateRequest(BaseModel):
    graph: GraphSchema
    input_dim: int
    batch_size: int = 32
    optimizer: str = "Adam"
    num_train_samples: int | None = None


class VramEstimateResponse(BaseModel):
    param_count: int
    effective_batch_size: int
    params_mb: float
    gradients_mb: float
    optimizer_mb: float
    activations_mb: float
    batch_data_mb: float
    total_mb: float
    available_mb: float | None = None
    fits: bool | None = None
