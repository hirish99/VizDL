"""GraphModule: compiles a DAG of ArchNodes into a trainable nn.Module.

Architecture nodes produce ArchRef objects during graph execution. GraphModel
traces the ArchRef DAG and builds a GraphModule whose forward() routes tensors
through the graph topology.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArchNode:
    """A node in the architecture blueprint (lazy — no actual computation)."""
    node_id: str
    module_type: str  # "Linear", "ReLU", "Split", "Concat", "DotProduct", "Add", "Identity", ...
    params: dict = field(default_factory=dict)
    inputs: list[ArchRef] = field(default_factory=list)
    n_outputs: int = 1


@dataclass
class ArchRef:
    """Reference to a specific output of an ArchNode."""
    node: ArchNode
    output_index: int = 0


# ---------------------------------------------------------------------------
# Graph tracing
# ---------------------------------------------------------------------------

def trace_graph(output_refs: list[ArchRef]) -> list[ArchNode]:
    """Walk backwards from output ArchRefs, return all ArchNodes in topo order."""
    seen: set[str] = set()
    all_nodes: dict[str, ArchNode] = {}

    def _visit(ref: ArchRef):
        node = ref.node
        if node.node_id in seen:
            return
        seen.add(node.node_id)
        all_nodes[node.node_id] = node
        for inp in node.inputs:
            _visit(inp)

    for ref in output_refs:
        _visit(ref)

    return _topo_sort(all_nodes)


def _topo_sort(nodes: dict[str, ArchNode]) -> list[ArchNode]:
    """Kahn's algorithm on ArchNode graph."""
    in_degree: dict[str, int] = {nid: 0 for nid in nodes}
    adj: dict[str, list[str]] = {nid: [] for nid in nodes}

    for node in nodes.values():
        for inp in node.inputs:
            src_id = inp.node.node_id
            if src_id in adj:
                adj[src_id].append(node.node_id)
                in_degree[node.node_id] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[ArchNode] = []
    while queue:
        nid = queue.popleft()
        order.append(nodes[nid])
        for succ in adj[nid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(order) != len(nodes):
        raise RuntimeError("Architecture graph contains a cycle")
    return order


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------

def infer_shapes_graph(nodes: list[ArchNode], input_dim: int | None):
    """Propagate tensor shapes through the architecture DAG.

    Mutates node.params in-place to fill missing dimensions.
    """
    shapes: dict[str, tuple] = {}  # node_id → tuple of per-output shapes

    for node in nodes:
        # Gather incoming shapes
        if not node.inputs:
            in_shapes = [input_dim] if input_dim is not None else [None]
        else:
            in_shapes = [
                shapes[ref.node.node_id][ref.output_index]
                for ref in node.inputs
            ]

        mt = node.module_type

        if mt == "Linear":
            if node.params.get("in_features") is None and in_shapes[0] is not None:
                node.params["in_features"] = in_shapes[0]
            out = node.params.get("out_features", in_shapes[0])
            shapes[node.node_id] = (out,)

        elif mt in ("ReLU", "GELU", "Sigmoid", "Tanh", "Identity", "Dropout"):
            shapes[node.node_id] = (in_shapes[0],)

        elif mt == "BatchNorm1d":
            if node.params.get("num_features") is None and in_shapes[0] is not None:
                node.params["num_features"] = in_shapes[0]
            shapes[node.node_id] = (in_shapes[0],)

        elif mt == "LayerNorm":
            if node.params.get("normalized_shape") is None and in_shapes[0] is not None:
                node.params["normalized_shape"] = in_shapes[0]
            shapes[node.node_id] = (in_shapes[0],)

        elif mt == "Split":
            sizes = node.params.get("split_sizes", [])
            shapes[node.node_id] = tuple(sizes)

        elif mt == "Concat":
            total = sum(s for s in in_shapes if s is not None)
            shapes[node.node_id] = (total if total else None,)

        elif mt == "DotProduct":
            shapes[node.node_id] = (1,)

        elif mt == "Add":
            shapes[node.node_id] = (in_shapes[0],)

        else:
            # Unknown type — assume passthrough
            shapes[node.node_id] = (in_shapes[0],)


# ---------------------------------------------------------------------------
# Module / op builders
# ---------------------------------------------------------------------------

MODULE_BUILDERS: dict[str, Callable[[dict], nn.Module]] = {
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


def _build_split_op(params: dict) -> Callable:
    sizes = list(params.get("split_sizes", []))
    dim = params.get("dim", -1)
    return lambda x: torch.split(x, sizes, dim=dim)


def _build_concat_op(params: dict) -> Callable:
    dim = params.get("dim", -1)
    return lambda *xs: torch.cat(xs, dim=dim)


def _build_dot_product_op(params: dict) -> Callable:
    return lambda a, b: torch.sum(a * b, dim=-1, keepdim=True)


def _build_add_op(params: dict) -> Callable:
    return lambda a, b: a + b


OP_BUILDERS: dict[str, Callable[[dict], Callable]] = {
    "Split": _build_split_op,
    "Concat": _build_concat_op,
    "DotProduct": _build_dot_product_op,
    "Add": _build_add_op,
}


# ---------------------------------------------------------------------------
# GraphModule
# ---------------------------------------------------------------------------

class GraphModule(nn.Module):
    """nn.Module that executes a DAG of operations in its forward pass."""

    def __init__(self, arch_nodes: list[ArchNode], output_refs: list[ArchRef]):
        super().__init__()

        self._topo_order = arch_nodes  # already topo-sorted
        self._output_refs = output_refs

        # Input nodes: arch nodes with no upstream inputs
        self._input_node_ids = [n.node_id for n in arch_nodes if not n.inputs]

        # Build edge map: node_id → list of (src_node_id, src_output_index)
        self._edge_map: dict[str, list[tuple[str, int]]] = {}
        for node in arch_nodes:
            self._edge_map[node.node_id] = [
                (ref.node.node_id, ref.output_index) for ref in node.inputs
            ]

        # Pure ops (no parameters — not registered as submodules)
        self._ops: dict[str, Callable] = {}

        # Register modules and ops
        for node in arch_nodes:
            if node.module_type in MODULE_BUILDERS:
                module = MODULE_BUILDERS[node.module_type](node.params)
                self.add_module(f"n_{node.node_id}", module)
            elif node.module_type in OP_BUILDERS:
                self._ops[node.node_id] = OP_BUILDERS[node.module_type](node.params)
            else:
                raise ValueError(f"Unknown architecture node type: {node.module_type}")

        # Store n_outputs per node for multi-output handling
        self._n_outputs = {n.node_id: n.n_outputs for n in arch_nodes}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values: dict[str, tuple] = {}  # node_id → tuple of outputs

        # Map raw input to input nodes
        if len(self._input_node_ids) == 1:
            raw_inputs = {self._input_node_ids[0]: x}
        else:
            # Multiple input nodes — x must be a tuple/list
            raw_inputs = {
                nid: x[i] for i, nid in enumerate(self._input_node_ids)
            }

        for node in self._topo_order:
            nid = node.node_id
            edges = self._edge_map[nid]

            # Gather inputs
            if not edges:
                # Input node — get from raw inputs
                inputs = [raw_inputs[nid]]
            else:
                inputs = [values[src_id][src_out] for src_id, src_out in edges]

            # Execute
            module = getattr(self, f"n_{nid}", None)
            op = self._ops.get(nid)

            if module is not None:
                result = module(inputs[0]) if len(inputs) == 1 else module(*inputs)
            elif op is not None:
                result = op(*inputs)
            else:
                result = inputs[0]  # fallback passthrough

            # Store as tuple of outputs
            if isinstance(result, tuple):
                values[nid] = result
            else:
                values[nid] = (result,)

        # Gather final outputs
        outputs = [values[ref.node.node_id][ref.output_index] for ref in self._output_refs]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
