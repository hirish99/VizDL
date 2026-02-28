"""Tests for structural operation nodes: Split, Concat, DotProduct, Add."""
import torch

from app.engine.graph_module import ArchNode, ArchRef, GraphModule, trace_graph, infer_shapes_graph
from app.nodes.structural import SplitNode, ConcatNode, DotProductNode, AddNode
from app.nodes.layers import LinearNode


def _make_upstream(node_id="upstream", out_features=64, in_features=4):
    """Create a simple Linear ArchRef for testing."""
    node = LinearNode()
    node._node_id = node_id
    return node.execute(in_features=in_features, out_features=out_features)[0]


class TestSplitNode:
    def test_execute(self):
        upstream = _make_upstream()
        node = SplitNode()
        node._node_id = "split1"
        result = node.execute(input=upstream, split_sizes="32,32")
        assert len(result) == 2
        assert isinstance(result[0], ArchRef)
        assert isinstance(result[1], ArchRef)
        assert result[0].node is result[1].node  # Same ArchNode
        assert result[0].output_index == 0
        assert result[1].output_index == 1
        assert result[0].node.module_type == "Split"
        assert result[0].node.params["split_sizes"] == [32, 32]

    def test_on_disable(self):
        upstream = _make_upstream()
        node = SplitNode()
        node._node_id = "split1"
        result = node.on_disable(input=upstream)
        assert len(result) == 2
        assert result[0] is upstream
        assert result[1] is upstream

    def test_custom_dim(self):
        upstream = _make_upstream()
        node = SplitNode()
        node._node_id = "split1"
        result = node.execute(input=upstream, split_sizes="16,48", dim=1)
        assert result[0].node.params["dim"] == 1
        assert result[0].node.params["split_sizes"] == [16, 48]


class TestConcatNode:
    def test_execute(self):
        a = _make_upstream("a", 32)
        b = _make_upstream("b", 32)
        node = ConcatNode()
        node._node_id = "cat1"
        result = node.execute(input_a=a, input_b=b)
        assert len(result) == 1
        assert isinstance(result[0], ArchRef)
        assert result[0].node.module_type == "Concat"
        assert len(result[0].node.inputs) == 2

    def test_on_disable(self):
        a = _make_upstream("a", 32)
        b = _make_upstream("b", 32)
        node = ConcatNode()
        node._node_id = "cat1"
        result = node.on_disable(input_a=a, input_b=b)
        assert len(result) == 1
        assert result[0] is a


class TestDotProductNode:
    def test_execute(self):
        a = _make_upstream("a", 8)
        b = _make_upstream("b", 8)
        node = DotProductNode()
        node._node_id = "dot1"
        result = node.execute(input_a=a, input_b=b)
        assert len(result) == 1
        assert result[0].node.module_type == "DotProduct"
        assert len(result[0].node.inputs) == 2

    def test_on_disable(self):
        a = _make_upstream("a", 8)
        b = _make_upstream("b", 8)
        node = DotProductNode()
        node._node_id = "dot1"
        result = node.on_disable(input_a=a, input_b=b)
        assert result[0] is a


class TestAddNode:
    def test_execute(self):
        a = _make_upstream("a", 8)
        b = _make_upstream("b", 8)
        node = AddNode()
        node._node_id = "add1"
        result = node.execute(input_a=a, input_b=b)
        assert len(result) == 1
        assert result[0].node.module_type == "Add"
        assert len(result[0].node.inputs) == 2

    def test_on_disable(self):
        a = _make_upstream("a", 8)
        b = _make_upstream("b", 8)
        node = AddNode()
        node._node_id = "add1"
        result = node.on_disable(input_a=a, input_b=b)
        assert result[0] is a


class TestStructuralIntegration:
    """Integration: structural nodes combined with layers, compiled into GraphModule."""

    def test_split_parallel_concat(self):
        """Linear → Split → [Linear, Linear] → Concat → Linear."""
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=64)[0]

        split = SplitNode(); split._node_id = "split"
        a, b = split.execute(input=ref, split_sizes="32,32")

        la = LinearNode(); la._node_id = "la"
        ra = la.execute(input=a, in_features=32, out_features=16)[0]

        lb = LinearNode(); lb._node_id = "lb"
        rb = lb.execute(input=b, in_features=32, out_features=16)[0]

        cat = ConcatNode(); cat._node_id = "cat"
        merged = cat.execute(input_a=ra, input_b=rb)[0]

        lout = LinearNode(); lout._node_id = "lout"
        out = lout.execute(input=merged, in_features=32, out_features=1)[0]

        nodes = trace_graph([out])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [out])

        x = torch.randn(5, 4)
        y = model(x)
        assert y.shape == (5, 1)

    def test_skip_connection_with_add(self):
        """Linear(4→8) → ReLU → Linear(8→8) → Add(+skip) → Linear(8→1)."""
        from app.nodes.layers import ReLUNode

        l1 = LinearNode(); l1._node_id = "l1"
        ref1 = l1.execute(in_features=4, out_features=8)[0]

        relu = ReLUNode(); relu._node_id = "relu"
        ref2 = relu.execute(input=ref1)[0]

        l2 = LinearNode(); l2._node_id = "l2"
        ref3 = l2.execute(input=ref2, in_features=8, out_features=8)[0]

        add = AddNode(); add._node_id = "add"
        ref4 = add.execute(input_a=ref3, input_b=ref1)[0]

        l3 = LinearNode(); l3._node_id = "l3"
        ref5 = l3.execute(input=ref4, in_features=8, out_features=1)[0]

        nodes = trace_graph([ref5])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [ref5])

        x = torch.randn(5, 4)
        y = model(x)
        assert y.shape == (5, 1)

    def test_dot_product_branches(self):
        """Linear → Split → [branch_a, branch_b] → DotProduct."""
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=64)[0]

        split = SplitNode(); split._node_id = "split"
        a, b = split.execute(input=ref, split_sizes="32,32")

        la = LinearNode(); la._node_id = "la"
        ra = la.execute(input=a, in_features=32, out_features=8)[0]

        lb = LinearNode(); lb._node_id = "lb"
        rb = lb.execute(input=b, in_features=32, out_features=8)[0]

        dot = DotProductNode(); dot._node_id = "dot"
        out = dot.execute(input_a=ra, input_b=rb)[0]

        nodes = trace_graph([out])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [out])

        x = torch.randn(5, 4)
        y = model(x)
        assert y.shape == (5, 1)
