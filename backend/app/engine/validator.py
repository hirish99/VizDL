"""Graph validation: cycle detection, type checking, required inputs."""
from collections import deque

from ..nodes.base import DataType, TYPE_COMPATIBILITY
from ..nodes.registry import NodeRegistry
from .graph import Graph


class ValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Graph validation failed: {errors}")


def validate_graph(graph: Graph) -> list[str]:
    """Validate a graph, returning a list of error messages (empty = valid)."""
    errors: list[str] = []
    errors.extend(_check_cycles(graph))
    errors.extend(_check_types(graph))
    errors.extend(_check_required_inputs(graph))
    return errors


def _check_cycles(graph: Graph) -> list[str]:
    """Detect cycles using Kahn's algorithm."""
    in_degree: dict[str, int] = {nid: 0 for nid in graph.nodes}
    adj: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.target_node in in_degree:
            in_degree[edge.target_node] += 1
        if edge.source_node in adj:
            adj[edge.source_node].append(edge.target_node)

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for succ in adj[node_id]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if visited != len(graph.nodes):
        return ["Graph contains a cycle"]
    return []


def _check_types(graph: Graph) -> list[str]:
    errors: list[str] = []
    for edge in graph.edges:
        src_node = graph.nodes.get(edge.source_node)
        tgt_node = graph.nodes.get(edge.target_node)
        if not src_node or not tgt_node:
            errors.append(f"Edge {edge.id} references missing node")
            continue

        try:
            src_cls = NodeRegistry.get(src_node.node_type)
            tgt_cls = NodeRegistry.get(tgt_node.node_type)
        except KeyError as e:
            errors.append(str(e))
            continue

        outputs = src_cls.RETURN_TYPES()
        if edge.source_output >= len(outputs):
            errors.append(
                f"Edge {edge.id}: source output index {edge.source_output} "
                f"out of range for {src_node.node_type}"
            )
            continue

        inputs = tgt_cls.INPUT_TYPES()
        if edge.target_input not in inputs:
            errors.append(
                f"Edge {edge.id}: target input '{edge.target_input}' "
                f"not found on {tgt_node.node_type}"
            )
            continue

        src_dtype = outputs[edge.source_output].dtype
        tgt_dtype = inputs[edge.target_input].dtype
        if tgt_dtype not in TYPE_COMPATIBILITY.get(src_dtype, set()):
            errors.append(
                f"Edge {edge.id}: type mismatch {src_dtype} → {tgt_dtype}"
            )

    return errors


def _check_required_inputs(graph: Graph) -> list[str]:
    errors: list[str] = []
    for node_id, node in graph.nodes.items():
        try:
            cls = NodeRegistry.get(node.node_type)
        except KeyError:
            errors.append(f"Unknown node type: {node.node_type}")
            continue

        inputs = cls.INPUT_TYPES()
        connected_inputs = {
            e.target_input for e in graph.get_incoming_edges(node_id)
        }

        for input_name, spec in inputs.items():
            if not spec.required:
                continue
            if spec.is_handle and input_name not in connected_inputs:
                # Check if it has a param value instead
                if input_name not in node.params:
                    errors.append(
                        f"Node '{node_id}' ({node.node_type}): "
                        f"required input '{input_name}' not connected"
                    )

    return errors
