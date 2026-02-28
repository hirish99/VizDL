"""Model assembly nodes: GraphModel (DAG) and legacy ModelAssembly (sequential)."""
from typing import Any

import torch
import torch.nn as nn

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry
from ..engine.graph_module import (
    ArchRef, GraphModule, trace_graph, infer_shapes_graph,
)


@NodeRegistry.register("GraphModel")
class GraphModelNode(BaseNode):
    CATEGORY = "Model"
    DISPLAY_NAME = "Graph Model"
    DESCRIPTION = "Compiles architecture graph into a trainable PyTorch model. Supports DAGs, skip connections, and parallel branches."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "architecture": InputSpec(dtype=DataType.ARCH, required=True),
            "dataset": InputSpec(dtype=DataType.DATASET, required=False, is_handle=False),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.MODEL, name="model")]

    def execute(self, **kwargs) -> tuple:
        arch_ref: ArchRef = kwargs["architecture"]
        input_dim = None
        dataset = kwargs.get("dataset")
        if dataset is not None:
            X = dataset.get("X") if isinstance(dataset, dict) else None
            if X is not None and hasattr(X, 'shape') and len(X.shape) >= 2:
                input_dim = X.shape[1]

        all_nodes = trace_graph([arch_ref])
        infer_shapes_graph(all_nodes, input_dim)

        # Validate required params before building
        for node in all_nodes:
            mt = node.module_type
            if mt == "Linear" and not node.params.get("in_features"):
                raise ValueError(
                    f"Linear node {node.node_id} is missing in_features. "
                    f"Connect a dataset to Graph Model or set it manually."
                )
            if mt == "BatchNorm1d" and not node.params.get("num_features"):
                raise ValueError(
                    f"BatchNorm1d node {node.node_id} is missing num_features. "
                    f"Set it manually or place it after a Linear node."
                )
            if mt == "LayerNorm" and not node.params.get("normalized_shape"):
                raise ValueError(
                    f"LayerNorm node {node.node_id} is missing normalized_shape. "
                    f"Set it manually or place it after a Linear node."
                )

        model = GraphModule(all_nodes, [arch_ref])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return (model,)


# ---------------------------------------------------------------------------
# Legacy: keep ModelAssembly for backward compatibility (deprecated)
# ---------------------------------------------------------------------------

LAYER_BUILDERS: dict[str, Any] = {
    "Linear": lambda p: nn.Linear(p["in_features"], p["out_features"], bias=p.get("bias", True)),
    "ReLU": lambda p: nn.ReLU(),
    "GELU": lambda p: nn.GELU(),
    "Sigmoid": lambda p: nn.Sigmoid(),
    "Tanh": lambda p: nn.Tanh(),
    "Dropout": lambda p: nn.Dropout(p=p.get("p", 0.5)),
    "BatchNorm1d": lambda p: nn.BatchNorm1d(p["num_features"]),
    "LayerNorm": lambda p: nn.LayerNorm(p["normalized_shape"]),
    "Identity": lambda p: nn.Identity(),
}


def _infer_shapes(specs: list[dict], input_dim: int | None = None) -> list[dict]:
    """Auto-fill in_features/num_features from previous layer's out_features."""
    last_out = input_dim
    for spec in specs:
        params = spec["params"]
        layer_type = spec["type"]
        if layer_type == "Linear":
            if params.get("in_features") is None and last_out is not None:
                params["in_features"] = last_out
            last_out = params.get("out_features", last_out)
        elif layer_type == "BatchNorm1d":
            if params.get("num_features") is None and last_out is not None:
                params["num_features"] = last_out
        elif layer_type == "LayerNorm":
            if params.get("normalized_shape") is None and last_out is not None:
                params["normalized_shape"] = last_out
    return specs


def build_sequential(specs: list[dict], input_dim: int | None = None) -> nn.Sequential:
    """Build nn.Sequential from layer specs (legacy, for backward compatibility)."""
    if not isinstance(specs, list):
        specs = [specs]
    flat: list[dict] = []
    for s in specs:
        if isinstance(s, list):
            flat.extend(s)
        else:
            flat.append(s)
    specs = flat
    specs = _infer_shapes(specs, input_dim=input_dim)
    layers: list[nn.Module] = []
    for i, spec in enumerate(specs):
        builder = LAYER_BUILDERS.get(spec["type"])
        if builder is None:
            raise ValueError(f"Unknown layer type: {spec['type']}")
        if spec["type"] == "Linear" and not spec["params"].get("in_features"):
            raise ValueError(
                f"Linear layer {i} is missing in_features. "
                f"Connect a dataset to Model Assembly or set it manually."
            )
        if spec["type"] == "BatchNorm1d" and not spec["params"].get("num_features"):
            raise ValueError(
                f"BatchNorm1d layer {i} is missing num_features. "
                f"Set it manually or place it after a Linear node."
            )
        if spec["type"] == "LayerNorm" and not spec["params"].get("normalized_shape"):
            raise ValueError(
                f"LayerNorm layer {i} is missing normalized_shape. "
                f"Set it manually or place it after a Linear node."
            )
        layers.append(builder(spec["params"]))
    return nn.Sequential(*layers)
