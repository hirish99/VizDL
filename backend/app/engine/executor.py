"""Execution engine: topological sort and execute pipeline."""
from collections import deque
from typing import Any, Callable

from ..nodes.base import BaseNode
from ..nodes.registry import NodeRegistry
from .cache import ExecutionCache
from .graph import Graph


ProgressCallback = Callable[[dict[str, Any]], None]


def topological_sort(graph: Graph) -> list[str]:
    """Kahn's algorithm returning node IDs in execution order."""
    in_degree: dict[str, int] = {nid: 0 for nid in graph.nodes}
    # Build adjacency list from edges (one entry per edge, not per unique successor)
    adj: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.target_node in in_degree:
            in_degree[edge.target_node] += 1
        if edge.source_node in adj:
            adj[edge.source_node].append(edge.target_node)

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for succ in adj[node_id]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(order) != len(graph.nodes):
        raise RuntimeError("Graph contains a cycle")
    return order


def execute_graph(
    graph: Graph,
    cache: ExecutionCache | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Execute all nodes in topological order, return {node_id: outputs}."""
    order = topological_sort(graph)
    results: dict[str, tuple[Any, ...]] = {}

    for node_id in order:
        node_inst = graph.nodes[node_id]
        node_cls = NodeRegistry.get(node_inst.node_type)
        node: BaseNode = node_cls()
        node._node_id = node_id  # Available for ArchNode construction

        # Resolve inputs from edges
        kwargs: dict[str, Any] = {}
        incoming = graph.get_incoming_edges(node_id)

        # Group edges by target_input for multi-input (e.g., layer specs)
        input_groups: dict[str, list] = {}
        for edge in incoming:
            src_results = results.get(edge.source_node)
            if src_results is None:
                continue
            value = src_results[edge.source_output]
            if edge.target_input not in input_groups:
                input_groups[edge.target_input] = []
            input_groups[edge.target_input].append((edge.order, value))

        for input_name, values in input_groups.items():
            values.sort(key=lambda x: x[0])
            if len(values) == 1:
                kwargs[input_name] = values[0][1]
            else:
                # Multiple edges to same input: collect as list
                kwargs[input_name] = [v for _, v in values]

        # Add params (properties panel values)
        for k, v in node_inst.params.items():
            if k not in kwargs:
                kwargs[k] = v

        # Inject progress callback if the node accepts it
        input_types = node_cls.INPUT_TYPES()
        if "progress_callback" in input_types:
            kwargs["progress_callback"] = progress_callback

        # Execute or ablate
        if node_inst.disabled:
            outputs = node.on_disable(**kwargs)
        else:
            outputs = node.execute(**kwargs)

        results[node_id] = outputs

        if progress_callback:
            progress_callback({
                "type": "node_complete",
                "node_id": node_id,
                "node_type": node_inst.node_type,
            })

    return results
