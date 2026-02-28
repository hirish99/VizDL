"""Tests for model building: GraphModelNode + legacy LAYER_BUILDERS."""
import pytest
import torch
import torch.nn as nn

from app.engine.graph_module import ArchNode, ArchRef, GraphModule, trace_graph, infer_shapes_graph
from app.nodes.model_assembly import GraphModelNode, LAYER_BUILDERS
from app.nodes.layers import LinearNode, ReLUNode, BatchNorm1dNode, LayerNormNode, GELUNode


def _device_of(model: nn.Module) -> torch.device:
    """Get the device the model is on."""
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Legacy layer builders (still used by GraphModule internally)
# ---------------------------------------------------------------------------

class TestLayerBuilders:
    def test_all_builders_exist(self):
        expected = {"Linear", "ReLU", "GELU", "Sigmoid", "Tanh", "Dropout", "BatchNorm1d", "LayerNorm", "Identity"}
        assert expected == set(LAYER_BUILDERS.keys())

    def test_identity_builder(self):
        layer = LAYER_BUILDERS["Identity"]({})
        assert isinstance(layer, nn.Identity)
        x = torch.randn(5, 8)
        assert torch.equal(layer(x), x)

    def test_gelu_builder(self):
        layer = LAYER_BUILDERS["GELU"]({})
        assert isinstance(layer, nn.GELU)

    def test_layernorm_builder(self):
        layer = LAYER_BUILDERS["LayerNorm"]({"normalized_shape": 8})
        assert isinstance(layer, nn.LayerNorm)
        x = torch.randn(5, 8)
        out = layer(x)
        assert out.shape == (5, 8)


# ---------------------------------------------------------------------------
# GraphModelNode tests
# ---------------------------------------------------------------------------

def _build_chain(*specs):
    """Build a linear chain of ArchRefs from (node_id, module_type, params) tuples."""
    ref = None
    for node_id, module_type, params in specs:
        inputs = [ref] if ref else []
        node = ArchNode(node_id, module_type, params, inputs)
        ref = ArchRef(node, 0)
    return ref


class TestGraphModelNormal:
    def test_basic_model(self):
        ref = _build_chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 8}),
            ("r1", "ReLU", {}),
            ("l2", "Linear", {"out_features": 1}),
        )
        node = GraphModelNode()
        node._node_id = "gm"
        model = node.execute(architecture=ref)[0]
        assert isinstance(model, GraphModule)
        device = _device_of(model)
        x = torch.randn(10, 4, device=device)
        out = model(x)
        assert out.shape == (10, 1)

    def test_with_dataset_infers_input_dim(self):
        ref = _build_chain(
            ("l1", "Linear", {"out_features": 8}),
            ("r1", "ReLU", {}),
            ("l2", "Linear", {"out_features": 1}),
        )
        dataset = {"X": torch.randn(100, 4), "y": torch.randn(100, 1)}
        node = GraphModelNode()
        node._node_id = "gm"
        model = node.execute(architecture=ref, dataset=dataset)[0]
        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 1)

    def test_missing_in_features_raises(self):
        ref = _build_chain(
            ("l1", "Linear", {"out_features": 8}),
        )
        node = GraphModelNode()
        node._node_id = "gm"
        with pytest.raises(ValueError, match="missing in_features"):
            node.execute(architecture=ref)

    def test_missing_batchnorm_num_features_raises(self):
        ref = _build_chain(
            ("bn", "BatchNorm1d", {}),
        )
        node = GraphModelNode()
        node._node_id = "gm"
        with pytest.raises(ValueError, match="missing num_features"):
            node.execute(architecture=ref)

    def test_missing_layernorm_shape_raises(self):
        ref = _build_chain(
            ("ln", "LayerNorm", {}),
        )
        node = GraphModelNode()
        node._node_id = "gm"
        with pytest.raises(ValueError, match="missing normalized_shape"):
            node.execute(architecture=ref)

    def test_layernorm_shape_inferred_from_linear(self):
        ref = _build_chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 8}),
            ("ln", "LayerNorm", {}),
            ("g1", "GELU", {}),
            ("l2", "Linear", {"out_features": 1}),
        )
        node = GraphModelNode()
        node._node_id = "gm"
        model = node.execute(architecture=ref)[0]
        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 1)


class TestGraphModelAblated:
    def test_identity_at_start(self):
        """[Identity, ReLU, Linear(?,1)] with dataset providing input_dim."""
        ref = _build_chain(
            ("id1", "Identity", {}),
            ("r1", "ReLU", {}),
            ("l1", "Linear", {"out_features": 1}),
        )
        dataset = {"X": torch.randn(100, 4), "y": torch.randn(100, 1)}
        node = GraphModelNode()
        node._node_id = "gm"
        model = node.execute(architecture=ref, dataset=dataset)[0]
        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 1)

    def test_identity_at_end(self):
        """[Linear(4,8), ReLU, Identity]"""
        ref = _build_chain(
            ("l1", "Linear", {"in_features": 4, "out_features": 8}),
            ("r1", "ReLU", {}),
            ("id1", "Identity", {}),
        )
        node = GraphModelNode()
        node._node_id = "gm"
        model = node.execute(architecture=ref)[0]
        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 8)

    def test_all_identity(self):
        """Model is pure passthrough — no parameters, stays on CPU."""
        ref = _build_chain(
            ("id1", "Identity", {}),
            ("id2", "Identity", {}),
        )
        dataset = {"X": torch.randn(100, 4), "y": torch.randn(100, 1)}
        node = GraphModelNode()
        node._node_id = "gm"
        model = node.execute(architecture=ref, dataset=dataset)[0]
        x = torch.randn(5, 4)
        out = model(x)
        assert torch.equal(out, x)

    def test_disabled_activation_omitted(self):
        """When activation is disabled (vanishes), chain is [Linear, Linear]."""
        # Build using actual nodes with ablation
        l1 = LinearNode(); l1._node_id = "l1"
        relu = ReLUNode(); relu._node_id = "relu"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.execute(in_features=4, out_features=8)[0]
        ref = relu.on_disable(input=ref)[0]  # vanishes
        ref = l2.execute(input=ref, out_features=1)[0]

        gm = GraphModelNode(); gm._node_id = "gm"
        model = gm.execute(architecture=ref)[0]
        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 1)

    def test_skip_connection_model(self):
        """Linear(4→8) → ReLU → Linear(8→8) → Add(+skip) → Linear(8→1)."""
        l1 = LinearNode(); l1._node_id = "l1"
        relu = ReLUNode(); relu._node_id = "relu"
        l2 = LinearNode(); l2._node_id = "l2"
        l3 = LinearNode(); l3._node_id = "l3"

        ref1 = l1.execute(in_features=4, out_features=8)[0]
        ref2 = relu.execute(input=ref1)[0]
        ref3 = l2.execute(input=ref2, in_features=8, out_features=8)[0]

        add_node = ArchNode("add", "Add", {}, [ref3, ref1])
        ref4 = ArchRef(add_node, 0)
        ref5 = l3.execute(input=ref4, in_features=8, out_features=1)[0]

        gm = GraphModelNode(); gm._node_id = "gm"
        model = gm.execute(architecture=ref5)[0]
        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 1)
