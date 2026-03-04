"""GraphModule: compiles a DAG of ArchNodes into a trainable nn.Module.

Architecture nodes produce ArchRef objects during graph execution. GraphModel
traces the ArchRef DAG and builds a GraphModule whose forward() routes tensors
through the graph topology.
"""
from __future__ import annotations

import math
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
# Shape helpers
# ---------------------------------------------------------------------------

def _shape_feat(shape):
    """Extract the feature dimension from a shape (int or (seq, feat) tuple)."""
    if isinstance(shape, tuple):
        return shape[1]
    return shape


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------

def infer_shapes_graph(nodes: list[ArchNode], input_dim: int | None):
    """Propagate tensor shapes through the architecture DAG.

    Mutates node.params in-place to fill missing dimensions.

    Shapes are tracked as either ``int`` (flat 2D tensor, last-dim size)
    or ``(seq_len, embed_dim)`` tuple (3D tensor from Tokenize).
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
            s = in_shapes[0]
            feat = _shape_feat(s) if s is not None else None
            if node.params.get("in_features") is None and feat is not None:
                node.params["in_features"] = feat
            out_feat = node.params.get("out_features", feat)
            if isinstance(s, tuple):
                shapes[node.node_id] = ((s[0], out_feat),)
            else:
                shapes[node.node_id] = (out_feat,)

        elif mt in ("ReLU", "GELU", "Sigmoid", "Tanh", "Identity", "Dropout"):
            shapes[node.node_id] = (in_shapes[0],)

        elif mt == "BatchNorm1d":
            if node.params.get("num_features") is None and in_shapes[0] is not None:
                node.params["num_features"] = _shape_feat(in_shapes[0])
            shapes[node.node_id] = (in_shapes[0],)

        elif mt == "LayerNorm":
            if node.params.get("normalized_shape") is None and in_shapes[0] is not None:
                node.params["normalized_shape"] = _shape_feat(in_shapes[0])
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

        elif mt == "Tokenize":
            n_tokens = node.params.get("n_tokens")
            s = in_shapes[0]
            if n_tokens and s is not None and isinstance(s, int):
                shapes[node.node_id] = ((n_tokens, s // n_tokens),)
            else:
                shapes[node.node_id] = ((n_tokens, None),)

        elif mt in ("SelfAttention", "CrossAttention"):
            # First input (query for cross-attn) determines output shape
            s = in_shapes[0]
            feat = _shape_feat(s) if s is not None else None
            if node.params.get("embed_dim") is None and feat is not None:
                node.params["embed_dim"] = feat
            shapes[node.node_id] = (in_shapes[0],)

        elif mt == "PositionalEncoding":
            s = in_shapes[0]
            feat = _shape_feat(s) if s is not None else None
            if node.params.get("embed_dim") is None and feat is not None:
                node.params["embed_dim"] = feat
            shapes[node.node_id] = (in_shapes[0],)

        elif mt == "Squeeze":
            s = in_shapes[0]
            mode = node.params.get("mode", "first")
            if isinstance(s, tuple):
                seq, dim = s
                if mode == "flatten":
                    shapes[node.node_id] = (seq * dim if seq and dim else None,)
                else:
                    shapes[node.node_id] = (dim,)
            else:
                shapes[node.node_id] = (s,)

        else:
            # Unknown type — assume passthrough
            shapes[node.node_id] = (in_shapes[0],)


# ---------------------------------------------------------------------------
# Attention module classes (used by MODULE_BUILDERS)
# ---------------------------------------------------------------------------

class _SelfAttentionModule(nn.Module):
    """Self-attention: Q=K=V=input."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


class _CrossAttentionModule(nn.Module):
    """Cross-attention: Q=query, K=V=context."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, context, context)
        return out


class _PositionalEncodingModule(nn.Module):
    """Sinusoidal positional encoding added to token embeddings."""
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:embed_dim // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


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
    "SelfAttention": lambda p: _SelfAttentionModule(p["embed_dim"], p["num_heads"], p.get("dropout", 0.0)),
    "CrossAttention": lambda p: _CrossAttentionModule(p["embed_dim"], p["num_heads"], p.get("dropout", 0.0)),
    "PositionalEncoding": lambda p: _PositionalEncodingModule(p.get("max_len", 256), p["embed_dim"]),
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


def _build_tokenize_op(params: dict) -> Callable:
    n_tokens = params["n_tokens"]
    return lambda x: x.view(x.shape[0], n_tokens, -1)


def _build_squeeze_op(params: dict) -> Callable:
    mode = params.get("mode", "first")
    if mode == "mean":
        return lambda x: x.mean(dim=1)
    elif mode == "flatten":
        return lambda x: x.flatten(1)
    else:  # "first"
        return lambda x: x[:, 0, :]


OP_BUILDERS: dict[str, Callable[[dict], Callable]] = {
    "Split": _build_split_op,
    "Concat": _build_concat_op,
    "DotProduct": _build_dot_product_op,
    "Add": _build_add_op,
    "Tokenize": _build_tokenize_op,
    "Squeeze": _build_squeeze_op,
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
