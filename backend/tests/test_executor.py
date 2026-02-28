"""Tests for the execution engine: topological sort and execute_graph."""
import pytest

from app.engine.executor import topological_sort, execute_graph
from app.engine.graph import Edge, Graph, NodeInstance
from app.engine.graph_module import ArchRef


class TestTopologicalSort:
    def test_simple_chain(self, simple_layer_graph):
        order = topological_sort(simple_layer_graph)
        assert order.index("linear1") < order.index("relu")
        assert order.index("relu") < order.index("linear2")

    def test_single_node(self):
        graph = Graph(
            nodes={"n1": NodeInstance(id="n1", node_type="Linear", params={"out_features": 1})},
            edges=[],
        )
        order = topological_sort(graph)
        assert order == ["n1"]

    def test_empty_graph(self):
        graph = Graph(nodes={}, edges=[])
        order = topological_sort(graph)
        assert order == []

    def test_cycle_raises(self):
        nodes = {
            "a": NodeInstance(id="a", node_type="ReLU"),
            "b": NodeInstance(id="b", node_type="ReLU"),
        }
        edges = [
            Edge(id="e1", source_node="a", source_output=0, target_node="b", target_input="input"),
            Edge(id="e2", source_node="b", source_output=0, target_node="a", target_input="input"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        with pytest.raises(RuntimeError, match="cycle"):
            topological_sort(graph)

    def test_diamond_graph(self):
        """A -> B, A -> C, B -> D, C -> D"""
        nodes = {
            "a": NodeInstance(id="a", node_type="Linear", params={"in_features": 4, "out_features": 8}),
            "b": NodeInstance(id="b", node_type="ReLU"),
            "c": NodeInstance(id="c", node_type="Sigmoid"),
            "d": NodeInstance(id="d", node_type="Linear", params={"out_features": 1}),
        }
        edges = [
            Edge(id="e1", source_node="a", source_output=0, target_node="b", target_input="input"),
            Edge(id="e2", source_node="a", source_output=0, target_node="c", target_input="input"),
            Edge(id="e3", source_node="b", source_output=0, target_node="d", target_input="input"),
            Edge(id="e4", source_node="c", source_output=0, target_node="d", target_input="input", order=1),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        order = topological_sort(graph)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_disabled_nodes_in_topo_order(self, simple_layer_graph):
        """Disabled nodes still appear in topological order."""
        simple_layer_graph.nodes["relu"].disabled = True
        order = topological_sort(simple_layer_graph)
        assert "relu" in order
        assert order.index("linear1") < order.index("relu")


class TestExecuteGraph:
    def test_simple_chain_execution(self, simple_layer_graph):
        results = execute_graph(simple_layer_graph)
        assert "linear1" in results
        assert "relu" in results
        assert "linear2" in results
        # Final output should be an ArchRef
        final = results["linear2"][0]
        assert isinstance(final, ArchRef)
        assert final.node.module_type == "Linear"

    def test_disabled_node_calls_on_disable(self, simple_layer_graph):
        """Disabling ReLU: it vanishes from the ArchRef graph."""
        simple_layer_graph.nodes["relu"].disabled = True
        results = execute_graph(simple_layer_graph)
        final = results["linear2"][0]
        assert isinstance(final, ArchRef)
        # Trace the graph to verify ReLU is gone
        from app.engine.graph_module import trace_graph
        nodes = trace_graph([final])
        types = [n.module_type for n in nodes]
        assert "ReLU" not in types
        assert types == ["Linear", "Linear"]

    def test_disabled_linear_inserts_identity(self, simple_layer_graph):
        """Disabling a Linear node replaces it with Identity."""
        simple_layer_graph.nodes["linear1"].disabled = True
        results = execute_graph(simple_layer_graph)
        final = results["linear2"][0]
        from app.engine.graph_module import trace_graph
        nodes = trace_graph([final])
        types = [n.module_type for n in nodes]
        assert types == ["Identity", "ReLU", "Linear"]

    def test_all_disabled(self, simple_layer_graph):
        """All nodes disabled."""
        for node in simple_layer_graph.nodes.values():
            node.disabled = True
        results = execute_graph(simple_layer_graph)
        final = results["linear2"][0]
        from app.engine.graph_module import trace_graph
        nodes = trace_graph([final])
        types = [n.module_type for n in nodes]
        assert types == ["Identity", "Identity"]

    def test_progress_callback_invoked(self, simple_layer_graph):
        messages = []
        execute_graph(simple_layer_graph, progress_callback=lambda m: messages.append(m))
        assert len(messages) == 3  # one per node
        assert all(m["type"] == "node_complete" for m in messages)

    def test_params_merged_with_edge_inputs(self):
        """Node params should be available alongside edge inputs."""
        graph = Graph(
            nodes={
                "linear1": NodeInstance(
                    id="linear1", node_type="Linear",
                    params={"in_features": 4, "out_features": 8, "bias": False},
                ),
            },
            edges=[],
        )
        results = execute_graph(graph)
        ref = results["linear1"][0]
        assert isinstance(ref, ArchRef)
        assert ref.node.params["bias"] is False
