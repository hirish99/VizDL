"""Graph data structures for the execution engine."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Edge:
    id: str
    source_node: str
    source_output: int  # index into RETURN_TYPES
    target_node: str
    target_input: str   # input name
    order: int = 0      # for ordering multiple inputs (e.g., layer specs)


@dataclass
class NodeInstance:
    id: str
    node_type: str
    params: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False
    position: dict[str, float] = field(default_factory=dict)


@dataclass
class Graph:
    nodes: dict[str, NodeInstance] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def get_incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target_node == node_id]

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source_node == node_id]

    def get_predecessors(self, node_id: str) -> set[str]:
        return {e.source_node for e in self.edges if e.target_node == node_id}

    def get_successors(self, node_id: str) -> set[str]:
        return {e.target_node for e in self.edges if e.source_node == node_id}
