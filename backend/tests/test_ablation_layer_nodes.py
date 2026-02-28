"""Tests for layer node on_disable (ablation) with ARCH-based system.

Layer nodes now produce ArchRef objects. Ablation behavior:
- Activations (ReLU, GELU, Sigmoid, Tanh) and Dropout: return upstream ArchRef
  directly (node vanishes from architecture graph).
- Linear, BatchNorm1d, LayerNorm: return ArchRef to Identity node.
"""
from app.engine.graph_module import ArchRef, ArchNode, trace_graph
from app.nodes.layers import (
    LinearNode, ReLUNode, GELUNode, SigmoidNode, TanhNode,
    DropoutNode, BatchNorm1dNode, LayerNormNode,
)


def _collect_types(ref: ArchRef) -> list[str]:
    """Trace ArchRef graph and return module types in topo (forward) order."""
    nodes = trace_graph([ref])
    return [n.module_type for n in nodes]


class TestLinearNodeAblation:
    def test_execute_returns_arch_ref(self):
        node = LinearNode(); node._node_id = "l1"
        result = node.execute(out_features=64)
        assert isinstance(result[0], ArchRef)
        assert result[0].node.module_type == "Linear"
        assert result[0].node.params["out_features"] == 64

    def test_on_disable_no_upstream(self):
        node = LinearNode(); node._node_id = "l1"
        result = node.on_disable()
        assert isinstance(result[0], ArchRef)
        assert result[0].node.module_type == "Identity"

    def test_on_disable_with_upstream(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]

        node = LinearNode(); node._node_id = "l1"
        result = node.on_disable(input=upstream_ref)
        assert result[0].node.module_type == "Identity"
        assert result[0].node.inputs[0] is upstream_ref

    def test_execute_vs_disable(self):
        node = LinearNode(); node._node_id = "l1"
        exec_result = node.execute(out_features=64)
        disable_result = node.on_disable()
        assert exec_result[0].node.module_type == "Linear"
        assert disable_result[0].node.module_type == "Identity"


class TestReLUNodeAblation:
    def test_on_disable_no_upstream(self):
        node = ReLUNode(); node._node_id = "r1"
        result = node.on_disable()
        assert isinstance(result[0], ArchRef)
        assert result[0].node.module_type == "Identity"

    def test_on_disable_passes_through_upstream(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]

        node = ReLUNode(); node._node_id = "r1"
        result = node.on_disable(input=upstream_ref)
        assert result[0] is upstream_ref

    def test_on_disable_vanishes_from_graph(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]

        node = ReLUNode(); node._node_id = "r1"
        result = node.on_disable(input=upstream_ref)
        types = _collect_types(result[0])
        assert types == ["Linear"]


class TestSigmoidNodeAblation:
    def test_on_disable_no_upstream(self):
        node = SigmoidNode(); node._node_id = "s1"
        result = node.on_disable()
        assert result[0].node.module_type == "Identity"

    def test_on_disable_passes_through(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]
        node = SigmoidNode(); node._node_id = "s1"
        result = node.on_disable(input=upstream_ref)
        assert result[0] is upstream_ref


class TestTanhNodeAblation:
    def test_on_disable_no_upstream(self):
        node = TanhNode(); node._node_id = "t1"
        result = node.on_disable()
        assert result[0].node.module_type == "Identity"

    def test_on_disable_passes_through(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]
        node = TanhNode(); node._node_id = "t1"
        result = node.on_disable(input=upstream_ref)
        assert result[0] is upstream_ref


class TestDropoutNodeAblation:
    def test_on_disable_no_upstream(self):
        node = DropoutNode(); node._node_id = "d1"
        result = node.on_disable()
        assert result[0].node.module_type == "Identity"

    def test_on_disable_passes_through(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]
        node = DropoutNode(); node._node_id = "d1"
        result = node.on_disable(input=upstream_ref)
        assert result[0] is upstream_ref

    def test_on_disable_ignores_p_param(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]
        node = DropoutNode(); node._node_id = "d1"
        result = node.on_disable(input=upstream_ref, p=0.9)
        assert result[0] is upstream_ref


class TestGELUNodeAblation:
    def test_on_disable_no_upstream(self):
        node = GELUNode(); node._node_id = "g1"
        result = node.on_disable()
        assert result[0].node.module_type == "Identity"

    def test_on_disable_passes_through(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]
        node = GELUNode(); node._node_id = "g1"
        result = node.on_disable(input=upstream_ref)
        assert result[0] is upstream_ref

    def test_execute_produces_gelu(self):
        node = GELUNode(); node._node_id = "g1"
        result = node.execute()
        assert result[0].node.module_type == "GELU"


class TestLayerNormNodeAblation:
    def test_on_disable_no_upstream(self):
        node = LayerNormNode(); node._node_id = "ln1"
        result = node.on_disable()
        assert result[0].node.module_type == "Identity"

    def test_on_disable_with_upstream(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]
        node = LayerNormNode(); node._node_id = "ln1"
        result = node.on_disable(input=upstream_ref)
        assert result[0].node.module_type == "Identity"
        assert result[0].node.inputs[0] is upstream_ref

    def test_execute_produces_layernorm(self):
        node = LayerNormNode(); node._node_id = "ln1"
        result = node.execute(normalized_shape=64)
        assert result[0].node.module_type == "LayerNorm"
        assert result[0].node.params["normalized_shape"] == 64

    def test_execute_normalized_shape_none(self):
        node = LayerNormNode(); node._node_id = "ln1"
        result = node.execute()
        assert result[0].node.params["normalized_shape"] is None


class TestBatchNorm1dNodeAblation:
    def test_on_disable_no_upstream(self):
        node = BatchNorm1dNode(); node._node_id = "bn1"
        result = node.on_disable()
        assert result[0].node.module_type == "Identity"

    def test_on_disable_with_upstream(self):
        upstream = LinearNode(); upstream._node_id = "l0"
        upstream_ref = upstream.execute(in_features=4, out_features=8)[0]
        node = BatchNorm1dNode(); node._node_id = "bn1"
        result = node.on_disable(input=upstream_ref)
        assert result[0].node.module_type == "Identity"
        assert result[0].node.inputs[0] is upstream_ref


class TestLayerAblationChaining:
    """Test ablation behavior in multi-layer chains via ArchRef graph."""

    def test_disabled_middle_activation(self):
        """Linear -> [disabled ReLU] -> Linear: ReLU vanishes."""
        l1 = LinearNode(); l1._node_id = "l1"
        relu = ReLUNode(); relu._node_id = "relu"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.execute(in_features=4, out_features=8)[0]
        ref = relu.on_disable(input=ref)[0]
        ref = l2.execute(input=ref, out_features=1)[0]

        types = _collect_types(ref)
        assert types == ["Linear", "Linear"]

    def test_disabled_first_linear(self):
        """[disabled Linear] -> ReLU -> Linear"""
        l1 = LinearNode(); l1._node_id = "l1"
        relu = ReLUNode(); relu._node_id = "relu"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.on_disable()[0]
        ref = relu.execute(input=ref)[0]
        ref = l2.execute(input=ref, out_features=1)[0]

        types = _collect_types(ref)
        assert types == ["Identity", "ReLU", "Linear"]

    def test_disabled_last_linear(self):
        """Linear -> ReLU -> [disabled Linear]"""
        l1 = LinearNode(); l1._node_id = "l1"
        relu = ReLUNode(); relu._node_id = "relu"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.execute(in_features=4, out_features=8)[0]
        ref = relu.execute(input=ref)[0]
        ref = l2.on_disable(input=ref)[0]

        types = _collect_types(ref)
        assert types == ["Linear", "ReLU", "Identity"]

    def test_all_layers_disabled(self):
        """[disabled Linear] -> [disabled ReLU] -> [disabled Linear]"""
        l1 = LinearNode(); l1._node_id = "l1"
        relu = ReLUNode(); relu._node_id = "relu"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.on_disable()[0]
        ref = relu.on_disable(input=ref)[0]  # passes through Identity
        ref = l2.on_disable(input=ref)[0]

        types = _collect_types(ref)
        assert types == ["Identity", "Identity"]

    def test_disabled_dropout_in_chain(self):
        """Linear -> [disabled Dropout] -> Linear"""
        l1 = LinearNode(); l1._node_id = "l1"
        dropout = DropoutNode(); dropout._node_id = "d1"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.execute(in_features=4, out_features=8)[0]
        ref = dropout.on_disable(input=ref)[0]
        ref = l2.execute(input=ref, out_features=1)[0]

        types = _collect_types(ref)
        assert types == ["Linear", "Linear"]

    def test_disabled_batchnorm_in_chain(self):
        """Linear -> [disabled BatchNorm] -> ReLU -> Linear"""
        l1 = LinearNode(); l1._node_id = "l1"
        bn = BatchNorm1dNode(); bn._node_id = "bn"
        relu = ReLUNode(); relu._node_id = "relu"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.execute(in_features=4, out_features=8)[0]
        ref = bn.on_disable(input=ref)[0]
        ref = relu.execute(input=ref)[0]
        ref = l2.execute(input=ref, out_features=1)[0]

        types = _collect_types(ref)
        # BatchNorm disabled → Identity, which is in the graph
        assert types == ["Linear", "Identity", "ReLU", "Linear"]

    def test_disabled_gelu_in_chain(self):
        """Linear -> [disabled GELU] -> Linear: GELU omitted."""
        l1 = LinearNode(); l1._node_id = "l1"
        gelu = GELUNode(); gelu._node_id = "gelu"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.execute(in_features=4, out_features=8)[0]
        ref = gelu.on_disable(input=ref)[0]
        ref = l2.execute(input=ref, out_features=1)[0]

        types = _collect_types(ref)
        assert types == ["Linear", "Linear"]

    def test_disabled_layernorm_in_chain(self):
        """Linear -> [disabled LayerNorm] -> GELU -> Linear"""
        l1 = LinearNode(); l1._node_id = "l1"
        ln = LayerNormNode(); ln._node_id = "ln"
        gelu = GELUNode(); gelu._node_id = "gelu"
        l2 = LinearNode(); l2._node_id = "l2"

        ref = l1.execute(in_features=4, out_features=8)[0]
        ref = ln.on_disable(input=ref)[0]
        ref = gelu.execute(input=ref)[0]
        ref = l2.execute(input=ref, out_features=1)[0]

        types = _collect_types(ref)
        # LayerNorm disabled → Identity, which stays in graph
        assert types == ["Linear", "Identity", "GELU", "Linear"]
