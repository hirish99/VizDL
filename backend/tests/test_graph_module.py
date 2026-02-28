"""Tests for GraphModule: ArchNode DAG compilation into nn.Module."""
import pytest
import torch
import torch.nn as nn

from app.engine.graph_module import (
    ArchNode, ArchRef, GraphModule,
    trace_graph, _topo_sort, infer_shapes_graph,
    MODULE_BUILDERS, OP_BUILDERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear(node_id, out, inputs=None, in_f=None):
    params = {"out_features": out}
    if in_f is not None:
        params["in_features"] = in_f
    n = ArchNode(node_id, "Linear", params, inputs or [])
    return ArchRef(n, 0)


def _relu(node_id, upstream):
    n = ArchNode(node_id, "ReLU", {}, [upstream])
    return ArchRef(n, 0)


def _identity(node_id, upstream=None):
    n = ArchNode(node_id, "Identity", {}, [upstream] if upstream else [])
    return ArchRef(n, 0)


def _add(node_id, a, b):
    n = ArchNode(node_id, "Add", {}, [a, b])
    return ArchRef(n, 0)


def _split(node_id, upstream, sizes):
    n = ArchNode(node_id, "Split", {"split_sizes": sizes, "dim": -1}, [upstream], n_outputs=2)
    return ArchRef(n, 0), ArchRef(n, 1)


def _concat(node_id, a, b):
    n = ArchNode(node_id, "Concat", {"dim": -1}, [a, b])
    return ArchRef(n, 0)


def _dot(node_id, a, b):
    n = ArchNode(node_id, "DotProduct", {}, [a, b])
    return ArchRef(n, 0)


# ---------------------------------------------------------------------------
# Tests: trace_graph and topo sort
# ---------------------------------------------------------------------------

class TestTraceGraph:
    def test_linear_chain(self):
        r1 = _linear("l1", 8, in_f=4)
        r2 = _relu("r1", r1)
        r3 = _linear("l2", 1, [r2])
        nodes = trace_graph([r3])
        ids = [n.node_id for n in nodes]
        assert ids == ["l1", "r1", "l2"]

    def test_diamond(self):
        """l1 → relu → l2 → add ← l1 (skip connection)."""
        r1 = _linear("l1", 8, in_f=4)
        r2 = _relu("r1", r1)
        r3 = _linear("l2", 8, [r2])
        r4 = _add("add", r3, r1)
        nodes = trace_graph([r4])
        ids = [n.node_id for n in nodes]
        assert ids.index("l1") < ids.index("r1")
        assert ids.index("r1") < ids.index("l2")
        assert ids.index("l2") < ids.index("add")
        assert len(ids) == 4

    def test_split_concat(self):
        r1 = _linear("l1", 64, in_f=4)
        a, b = _split("s1", r1, [32, 32])
        r2 = _linear("la", 16, [a])
        r3 = _linear("lb", 16, [b])
        cat = _concat("c1", r2, r3)
        nodes = trace_graph([cat])
        ids = [n.node_id for n in nodes]
        assert ids[0] == "l1"
        assert ids[-1] == "c1"
        assert "s1" in ids

    def test_single_node(self):
        r1 = _linear("l1", 8, in_f=4)
        nodes = trace_graph([r1])
        assert len(nodes) == 1
        assert nodes[0].node_id == "l1"


class TestTopoSort:
    def test_cycle_raises(self):
        """Manually create a cycle."""
        a = ArchNode("a", "ReLU", {}, [])
        b = ArchNode("b", "ReLU", {}, [ArchRef(a, 0)])
        # Create cycle: a depends on b
        a.inputs = [ArchRef(b, 0)]
        with pytest.raises(RuntimeError, match="cycle"):
            _topo_sort({"a": a, "b": b})


# ---------------------------------------------------------------------------
# Tests: shape inference
# ---------------------------------------------------------------------------

class TestInferShapesGraph:
    def test_linear_chain(self):
        r1 = _linear("l1", 8, in_f=4)
        r2 = _relu("r1", r1)
        r3 = _linear("l2", 1, [r2])
        nodes = trace_graph([r3])
        infer_shapes_graph(nodes, input_dim=4)
        l2 = [n for n in nodes if n.node_id == "l2"][0]
        assert l2.params["in_features"] == 8

    def test_input_dim_fills_first_linear(self):
        r1 = _linear("l1", 8)  # in_features not set
        nodes = trace_graph([r1])
        infer_shapes_graph(nodes, input_dim=4)
        assert nodes[0].params["in_features"] == 4

    def test_no_input_dim_leaves_none(self):
        r1 = _linear("l1", 8)
        nodes = trace_graph([r1])
        infer_shapes_graph(nodes, input_dim=None)
        assert nodes[0].params.get("in_features") is None

    def test_batchnorm_after_linear(self):
        r1 = _linear("l1", 8, in_f=4)
        bn = ArchNode("bn", "BatchNorm1d", {}, [r1])
        ref = ArchRef(bn, 0)
        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        assert bn.params["num_features"] == 8

    def test_layernorm_after_linear(self):
        r1 = _linear("l1", 16, in_f=4)
        ln = ArchNode("ln", "LayerNorm", {}, [r1])
        ref = ArchRef(ln, 0)
        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        assert ln.params["normalized_shape"] == 16

    def test_split_shapes(self):
        r1 = _linear("l1", 64, in_f=4)
        a, b = _split("s1", r1, [32, 32])
        nodes = trace_graph([a])  # a and b share same ArchNode
        infer_shapes_graph(nodes, input_dim=4)
        # Split node should have shapes (32, 32)

    def test_concat_shapes(self):
        r1 = _linear("la", 16, in_f=4)
        r2 = _linear("lb", 8, in_f=4)
        cat = _concat("c1", r1, r2)
        nodes = trace_graph([cat])
        infer_shapes_graph(nodes, input_dim=4)
        # Concat output shape should be 16 + 8 = 24

    def test_skip_connection_shapes(self):
        """Linear(4→8) → ReLU → Linear(8→8) → Add ← skip from first Linear."""
        r1 = _linear("l1", 8, in_f=4)
        r2 = _relu("r1", r1)
        r3 = _linear("l2", 8, [r2])
        r4 = _add("add", r3, r1)
        r5 = _linear("l3", 1, [r4])
        nodes = trace_graph([r5])
        infer_shapes_graph(nodes, input_dim=4)
        l2 = [n for n in nodes if n.node_id == "l2"][0]
        l3 = [n for n in nodes if n.node_id == "l3"][0]
        assert l2.params["in_features"] == 8
        assert l3.params["in_features"] == 8


# ---------------------------------------------------------------------------
# Tests: module/op builders
# ---------------------------------------------------------------------------

class TestBuilders:
    def test_all_module_builders_exist(self):
        expected = {"Linear", "ReLU", "GELU", "Sigmoid", "Tanh", "Dropout",
                    "BatchNorm1d", "LayerNorm", "Identity"}
        assert expected == set(MODULE_BUILDERS.keys())

    def test_all_op_builders_exist(self):
        expected = {"Split", "Concat", "DotProduct", "Add"}
        assert expected == set(OP_BUILDERS.keys())


# ---------------------------------------------------------------------------
# Tests: GraphModule forward pass
# ---------------------------------------------------------------------------

class TestGraphModuleForward:
    def test_linear_chain(self):
        """Simple: Linear(4→8) → ReLU → Linear(8→1)."""
        r1 = _linear("l1", 8, in_f=4)
        r2 = _relu("r1", r1)
        r3 = _linear("l2", 1, [r2])
        nodes = trace_graph([r3])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [r3])
        x = torch.randn(5, 4)
        out = model(x)
        assert out.shape == (5, 1)

    def test_skip_connection(self):
        """Linear(4→8) → ReLU → Linear(8→8) → Add(+skip from L1) → Linear(8→1)."""
        r1 = _linear("l1", 8, in_f=4)
        r2 = _relu("r1", r1)
        r3 = _linear("l2", 8, [r2])
        r4 = _add("add", r3, r1)
        r5 = _linear("l3", 1, [r4])
        nodes = trace_graph([r5])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [r5])
        x = torch.randn(5, 4)
        out = model(x)
        assert out.shape == (5, 1)

    def test_split_parallel_concat(self):
        """Linear(4→64) → Split(32,32) → [Linear(32→16), Linear(32→16)] → Concat → Linear(32→1)."""
        r1 = _linear("l1", 64, in_f=4)
        a, b = _split("s1", r1, [32, 32])
        ra = _linear("la", 16, [a], in_f=32)
        rb = _linear("lb", 16, [b], in_f=32)
        cat = _concat("c1", ra, rb)
        out_ref = _linear("lout", 1, [cat], in_f=32)
        nodes = trace_graph([out_ref])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [out_ref])
        x = torch.randn(5, 4)
        out = model(x)
        assert out.shape == (5, 1)

    def test_dot_product(self):
        """Linear(4→64) → Split(32,32) → [Linear(32→8), Linear(32→8)] → DotProduct → output is (batch, 1)."""
        r1 = _linear("l1", 64, in_f=4)
        a, b = _split("s1", r1, [32, 32])
        ra = _linear("la", 8, [a], in_f=32)
        rb = _linear("lb", 8, [b], in_f=32)
        dot = _dot("dp", ra, rb)
        nodes = trace_graph([dot])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [dot])
        x = torch.randn(5, 4)
        out = model(x)
        assert out.shape == (5, 1)

    def test_identity_passthrough(self):
        """Identity → Linear: input passes through Identity unchanged."""
        r1 = _identity("id1")
        r2 = _linear("l1", 1, [r1], in_f=4)
        nodes = trace_graph([r2])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [r2])
        x = torch.randn(5, 4)
        out = model(x)
        assert out.shape == (5, 1)

    def test_model_has_parameters(self):
        """GraphModule registers submodules correctly."""
        r1 = _linear("l1", 8, in_f=4)
        r2 = _linear("l2", 1, [r1], in_f=8)
        nodes = trace_graph([r2])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [r2])
        params = list(model.parameters())
        assert len(params) > 0

    def test_gradient_flows(self):
        """Verify gradients flow through the DAG."""
        r1 = _linear("l1", 8, in_f=4)
        r2 = _relu("r1", r1)
        r3 = _linear("l2", 1, [r2])
        nodes = trace_graph([r3])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [r3])
        x = torch.randn(5, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_ablated_model_identity_chain(self):
        """Identity → Identity → Linear: ablated nodes become Identity."""
        r1 = _identity("id1")
        r2 = _identity("id2", r1)
        r3 = _linear("l1", 1, [r2], in_f=4)
        nodes = trace_graph([r3])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [r3])
        x = torch.randn(5, 4)
        out = model(x)
        assert out.shape == (5, 1)
