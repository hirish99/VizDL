"""Tests for shape inference: DAG-based infer_shapes_graph + legacy _infer_shapes."""
import torch

from app.engine.graph_module import (
    ArchNode, ArchRef, infer_shapes_graph, trace_graph,
)
from app.nodes.model_assembly import _infer_shapes


# ---------------------------------------------------------------------------
# DAG shape inference (new system)
# ---------------------------------------------------------------------------

def _chain(*specs):
    """Build a linear ArchNode chain from (id, type, params) tuples, return nodes list."""
    ref = None
    for node_id, module_type, params in specs:
        inputs = [ref] if ref else []
        node = ArchNode(node_id, module_type, params, inputs)
        ref = ArchRef(node, 0)
    return trace_graph([ref])


class TestInferShapesGraphNormal:
    def test_basic_linear_chain(self):
        nodes = _chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 8}),
            ("r1", "ReLU", {}),
            ("l2", "Linear", {"out_features": 1}),
        )
        infer_shapes_graph(nodes, input_dim=4)
        l2 = [n for n in nodes if n.node_id == "l2"][0]
        assert l2.params["in_features"] == 8

    def test_input_dim_fills_first_linear(self):
        nodes = _chain(("l1", "Linear", {"out_features": 8}),)
        infer_shapes_graph(nodes, input_dim=4)
        assert nodes[0].params["in_features"] == 4

    def test_no_input_dim_leaves_none(self):
        nodes = _chain(("l1", "Linear", {"out_features": 8}),)
        infer_shapes_graph(nodes, input_dim=None)
        assert nodes[0].params.get("in_features") is None

    def test_batchnorm_after_linear(self):
        nodes = _chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 8}),
            ("bn", "BatchNorm1d", {}),
        )
        infer_shapes_graph(nodes, input_dim=4)
        bn = [n for n in nodes if n.node_id == "bn"][0]
        assert bn.params["num_features"] == 8

    def test_layernorm_after_linear(self):
        nodes = _chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 16}),
            ("ln", "LayerNorm", {}),
        )
        infer_shapes_graph(nodes, input_dim=4)
        ln = [n for n in nodes if n.node_id == "ln"][0]
        assert ln.params["normalized_shape"] == 16

    def test_multiple_linears(self):
        nodes = _chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 32}),
            ("r1", "ReLU", {}),
            ("l2", "Linear", {"out_features": 16}),
            ("r2", "ReLU", {}),
            ("l3", "Linear", {"out_features": 1}),
        )
        infer_shapes_graph(nodes, input_dim=4)
        l2 = [n for n in nodes if n.node_id == "l2"][0]
        l3 = [n for n in nodes if n.node_id == "l3"][0]
        assert l2.params["in_features"] == 32
        assert l3.params["in_features"] == 16


class TestInferShapesGraphAblated:
    def test_ablated_first_linear_identity(self):
        """Identity at start: shape comes from input_dim."""
        nodes = _chain(
            ("id1", "Identity", {}),
            ("r1", "ReLU", {}),
            ("l1", "Linear", {"out_features": 1}),
        )
        infer_shapes_graph(nodes, input_dim=4)
        l1 = [n for n in nodes if n.node_id == "l1"][0]
        assert l1.params["in_features"] == 4

    def test_ablated_middle_activation_removed(self):
        """When ReLU vanishes (disabled), chain is Linear→Linear."""
        nodes = _chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 8}),
            ("l2", "Linear", {"out_features": 1}),
        )
        infer_shapes_graph(nodes, input_dim=4)
        l2 = [n for n in nodes if n.node_id == "l2"][0]
        assert l2.params["in_features"] == 8

    def test_identity_preserves_shape_through_chain(self):
        """Multiple identities don't break shape tracking."""
        nodes = _chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 16}),
            ("id1", "Identity", {}),
            ("id2", "Identity", {}),
            ("l2", "Linear", {"out_features": 1}),
        )
        infer_shapes_graph(nodes, input_dim=4)
        l2 = [n for n in nodes if n.node_id == "l2"][0]
        assert l2.params["in_features"] == 16

    def test_batchnorm_after_identity_uses_input_dim(self):
        nodes = _chain(
            ("id1", "Identity", {}),
            ("bn", "BatchNorm1d", {}),
        )
        infer_shapes_graph(nodes, input_dim=8)
        bn = [n for n in nodes if n.node_id == "bn"][0]
        assert bn.params["num_features"] == 8

    def test_batchnorm_after_identity_no_input_dim(self):
        nodes = _chain(
            ("id1", "Identity", {}),
            ("bn", "BatchNorm1d", {}),
        )
        infer_shapes_graph(nodes, input_dim=None)
        bn = [n for n in nodes if n.node_id == "bn"][0]
        assert bn.params.get("num_features") is None


class TestInferShapesGraphDAG:
    """Shape inference on non-linear (DAG) topologies."""

    def test_skip_connection(self):
        """L1(4→8) → ReLU → L2(→8) → Add(+L1) → L3(→1)."""
        l1 = ArchNode("l1", "Linear", {"in_features": 4, "out_features": 8}, [])
        r1 = ArchNode("r1", "ReLU", {}, [ArchRef(l1, 0)])
        l2 = ArchNode("l2", "Linear", {"out_features": 8}, [ArchRef(r1, 0)])
        add = ArchNode("add", "Add", {}, [ArchRef(l2, 0), ArchRef(l1, 0)])
        l3 = ArchNode("l3", "Linear", {"out_features": 1}, [ArchRef(add, 0)])

        nodes = trace_graph([ArchRef(l3, 0)])
        infer_shapes_graph(nodes, input_dim=4)

        assert l2.params["in_features"] == 8
        assert l3.params["in_features"] == 8

    def test_split_concat(self):
        """L1(4→64) → Split(32,32) → [La(→16), Lb(→16)] → Concat → Lout(→1)."""
        l1 = ArchNode("l1", "Linear", {"in_features": 4, "out_features": 64}, [])
        split = ArchNode("s1", "Split", {"split_sizes": [32, 32], "dim": -1},
                         [ArchRef(l1, 0)], n_outputs=2)
        la = ArchNode("la", "Linear", {"out_features": 16}, [ArchRef(split, 0)])
        lb = ArchNode("lb", "Linear", {"out_features": 16}, [ArchRef(split, 1)])
        cat = ArchNode("c1", "Concat", {"dim": -1}, [ArchRef(la, 0), ArchRef(lb, 0)])
        lout = ArchNode("lout", "Linear", {"out_features": 1}, [ArchRef(cat, 0)])

        nodes = trace_graph([ArchRef(lout, 0)])
        infer_shapes_graph(nodes, input_dim=4)

        assert la.params["in_features"] == 32
        assert lb.params["in_features"] == 32
        assert lout.params["in_features"] == 32  # 16 + 16

    def test_dot_product(self):
        """DotProduct output shape is 1."""
        l1 = ArchNode("l1", "Linear", {"in_features": 4, "out_features": 64}, [])
        split = ArchNode("s1", "Split", {"split_sizes": [32, 32], "dim": -1},
                         [ArchRef(l1, 0)], n_outputs=2)
        la = ArchNode("la", "Linear", {"out_features": 8}, [ArchRef(split, 0)])
        lb = ArchNode("lb", "Linear", {"out_features": 8}, [ArchRef(split, 1)])
        dot = ArchNode("dp", "DotProduct", {}, [ArchRef(la, 0), ArchRef(lb, 0)])
        lout = ArchNode("lout", "Linear", {"out_features": 1}, [ArchRef(dot, 0)])

        nodes = trace_graph([ArchRef(lout, 0)])
        infer_shapes_graph(nodes, input_dim=4)

        assert la.params["in_features"] == 32
        assert lb.params["in_features"] == 32
        assert lout.params["in_features"] == 1  # DotProduct outputs 1


# ---------------------------------------------------------------------------
# Legacy shape inference (backward compat)
# ---------------------------------------------------------------------------

class TestLegacyInferShapes:
    def test_basic_linear_chain(self):
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "ReLU", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        result = _infer_shapes(specs, input_dim=4)
        assert result[2]["params"]["in_features"] == 8

    def test_input_dim_fills_first_linear(self):
        specs = [{"type": "Linear", "params": {"in_features": None, "out_features": 8}}]
        result = _infer_shapes(specs, input_dim=4)
        assert result[0]["params"]["in_features"] == 4

    def test_batchnorm_after_linear(self):
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "BatchNorm1d", "params": {"num_features": None}},
        ]
        result = _infer_shapes(specs)
        assert result[1]["params"]["num_features"] == 8
