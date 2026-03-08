"""Tests for graph validation: cycles, type checking, required inputs."""
from app.engine.graph import Edge, Graph, NodeInstance
from app.engine.validator import validate_graph


class TestCycleDetection:
    def test_no_cycle(self, simple_layer_graph):
        errors = validate_graph(simple_layer_graph)
        assert not any("cycle" in e.lower() for e in errors)

    def test_self_loop(self):
        graph = Graph(
            nodes={"a": NodeInstance(id="a", node_type="ReLU")},
            edges=[Edge(id="e1", source_node="a", source_output=0,
                        target_node="a", target_input="input")],
        )
        errors = validate_graph(graph)
        assert any("cycle" in e.lower() for e in errors)

    def test_two_node_cycle(self):
        graph = Graph(
            nodes={
                "a": NodeInstance(id="a", node_type="ReLU"),
                "b": NodeInstance(id="b", node_type="ReLU"),
            },
            edges=[
                Edge(id="e1", source_node="a", source_output=0,
                     target_node="b", target_input="input"),
                Edge(id="e2", source_node="b", source_output=0,
                     target_node="a", target_input="input"),
            ],
        )
        errors = validate_graph(graph)
        assert any("cycle" in e.lower() for e in errors)

    def test_no_cycle_in_dag(self):
        """A -> B, A -> C, B -> D, C -> D (diamond, no cycle)."""
        graph = Graph(
            nodes={
                "a": NodeInstance(id="a", node_type="Linear", params={"out_features": 8}),
                "b": NodeInstance(id="b", node_type="ReLU"),
                "c": NodeInstance(id="c", node_type="Sigmoid"),
                "d": NodeInstance(id="d", node_type="Linear", params={"out_features": 1}),
            },
            edges=[
                Edge(id="e1", source_node="a", source_output=0, target_node="b", target_input="input"),
                Edge(id="e2", source_node="a", source_output=0, target_node="c", target_input="input"),
                Edge(id="e3", source_node="b", source_output=0, target_node="d", target_input="input"),
                Edge(id="e4", source_node="c", source_output=0, target_node="d", target_input="input", order=1),
            ],
        )
        errors = validate_graph(graph)
        assert not any("cycle" in e.lower() for e in errors)


class TestTypeChecking:
    def test_valid_types(self, simple_layer_graph):
        errors = validate_graph(simple_layer_graph)
        assert not any("type mismatch" in e.lower() for e in errors)

    def test_missing_source_node(self):
        graph = Graph(
            nodes={"b": NodeInstance(id="b", node_type="ReLU")},
            edges=[Edge(id="e1", source_node="missing", source_output=0,
                        target_node="b", target_input="input")],
        )
        errors = validate_graph(graph)
        assert any("missing node" in e.lower() for e in errors)

    def test_invalid_output_index(self):
        graph = Graph(
            nodes={
                "a": NodeInstance(id="a", node_type="ReLU"),
                "b": NodeInstance(id="b", node_type="ReLU"),
            },
            edges=[Edge(id="e1", source_node="a", source_output=99,
                        target_node="b", target_input="input")],
        )
        errors = validate_graph(graph)
        assert any("out of range" in e.lower() for e in errors)

    def test_invalid_input_name(self):
        graph = Graph(
            nodes={
                "a": NodeInstance(id="a", node_type="ReLU"),
                "b": NodeInstance(id="b", node_type="ReLU"),
            },
            edges=[Edge(id="e1", source_node="a", source_output=0,
                        target_node="b", target_input="nonexistent_input")],
        )
        errors = validate_graph(graph)
        assert any("not found" in e.lower() for e in errors)


class TestRequiredInputs:
    def test_required_input_connected(self, simple_layer_graph):
        errors = validate_graph(simple_layer_graph)
        # Linear.out_features is required but is_handle=False, so it's set via params
        # input is not required (required=False)
        assert not any("required input" in e.lower() for e in errors)

    def test_required_input_missing(self):
        """Linear's out_features is required and is_handle=False, but has default in params."""
        graph = Graph(
            nodes={
                "a": NodeInstance(id="a", node_type="Linear", params={"out_features": 8}),
                # b requires model input (is_handle=True, required=True) but nothing is connected
                "b": NodeInstance(id="b", node_type="Adam"),
            },
            edges=[],
        )
        errors = validate_graph(graph)
        # Adam requires model input which is not connected
        assert any("required input" in e.lower() and "model" in e.lower() for e in errors)


class TestEdgeDeletion:
    """Verify validation behavior when edges are removed (simulating UI edge deletion)."""

    def test_deleted_edge_cross_attention_missing_context(self):
        """Removing input_context edge from CrossAttention should fail validation."""
        graph = Graph(
            nodes={
                "embed": NodeInstance(id="embed", node_type="Linear", params={"out_features": 64}),
                "trunk": NodeInstance(id="trunk", node_type="Linear", params={"out_features": 64}),
                "cross": NodeInstance(id="cross", node_type="CrossAttention", params={"num_heads": 4}),
            },
            edges=[
                Edge(id="e1", source_node="embed", source_output=0,
                     target_node="cross", target_input="input_query"),
                # input_context edge deleted — not present
            ],
        )
        errors = validate_graph(graph)
        assert any("required input" in e.lower() and "input_context" in e.lower() for e in errors)

    def test_deleted_edge_still_valid_standalone_nodes(self):
        """Removing an edge leaving two standalone nodes should still validate."""
        graph = Graph(
            nodes={
                "a": NodeInstance(id="a", node_type="ReLU"),
                "b": NodeInstance(id="b", node_type="ReLU"),
            },
            edges=[],  # Edge between a→b deleted
        )
        errors = validate_graph(graph)
        assert not any("cycle" in e.lower() for e in errors)
        assert not any("required input" in e.lower() for e in errors)


class TestValidationWithDisabledNodes:
    def test_disabled_node_still_validates(self, simple_layer_graph):
        """Disabled nodes pass validation — disabled is execution-time, not validation-time."""
        simple_layer_graph.nodes["relu"].disabled = True
        errors = validate_graph(simple_layer_graph)
        assert not any("cycle" in e.lower() for e in errors)
