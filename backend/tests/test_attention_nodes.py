"""Tests for attention nodes: Tokenize, PositionalEncoding, SelfAttention, CrossAttention, Squeeze."""
import torch
import pytest

from app.engine.graph_module import (
    ArchNode, ArchRef, GraphModule, trace_graph, infer_shapes_graph,
    _SelfAttentionModule, _CrossAttentionModule, _PositionalEncodingModule,
)
from app.nodes.attention import (
    TokenizeNode, PositionalEncodingNode, SelfAttentionNode,
    CrossAttentionNode, SqueezeNode,
)
from app.nodes.layers import LinearNode
from app.nodes.structural import SplitNode


def _make_upstream(node_id="upstream", out_features=64, in_features=4):
    node = LinearNode()
    node._node_id = node_id
    return node.execute(in_features=in_features, out_features=out_features)[0]


# ---------------------------------------------------------------------------
# Unit tests: node execute / on_disable
# ---------------------------------------------------------------------------

class TestTokenizeNode:
    def test_execute(self):
        upstream = _make_upstream()
        node = TokenizeNode()
        node._node_id = "tok1"
        result = node.execute(input=upstream, n_tokens=64)
        assert len(result) == 1
        assert isinstance(result[0], ArchRef)
        assert result[0].node.module_type == "Tokenize"
        assert result[0].node.params["n_tokens"] == 64

    def test_on_disable(self):
        upstream = _make_upstream()
        node = TokenizeNode()
        node._node_id = "tok1"
        result = node.on_disable(input=upstream)
        assert result[0] is upstream


class TestPositionalEncodingNode:
    def test_execute(self):
        upstream = _make_upstream()
        node = PositionalEncodingNode()
        node._node_id = "pe1"
        result = node.execute(input=upstream, max_len=128)
        assert result[0].node.module_type == "PositionalEncoding"
        assert result[0].node.params["max_len"] == 128

    def test_execute_default_max_len(self):
        upstream = _make_upstream()
        node = PositionalEncodingNode()
        node._node_id = "pe1"
        result = node.execute(input=upstream)
        assert result[0].node.params["max_len"] == 256

    def test_on_disable(self):
        upstream = _make_upstream()
        node = PositionalEncodingNode()
        node._node_id = "pe1"
        result = node.on_disable(input=upstream)
        assert result[0] is upstream


class TestSelfAttentionNode:
    def test_execute(self):
        upstream = _make_upstream()
        node = SelfAttentionNode()
        node._node_id = "sa1"
        result = node.execute(input=upstream, num_heads=4, dropout=0.1)
        assert result[0].node.module_type == "SelfAttention"
        assert result[0].node.params["num_heads"] == 4
        assert result[0].node.params["dropout"] == 0.1

    def test_on_disable(self):
        upstream = _make_upstream()
        node = SelfAttentionNode()
        node._node_id = "sa1"
        result = node.on_disable(input=upstream)
        assert result[0] is upstream


class TestCrossAttentionNode:
    def test_execute(self):
        a = _make_upstream("a", 32)
        b = _make_upstream("b", 32)
        node = CrossAttentionNode()
        node._node_id = "ca1"
        result = node.execute(input_query=a, input_context=b, num_heads=4)
        assert len(result) == 1
        assert result[0].node.module_type == "CrossAttention"
        assert len(result[0].node.inputs) == 2

    def test_on_disable(self):
        a = _make_upstream("a", 32)
        b = _make_upstream("b", 32)
        node = CrossAttentionNode()
        node._node_id = "ca1"
        result = node.on_disable(input_query=a, input_context=b)
        assert result[0] is a


class TestSqueezeNode:
    def test_execute(self):
        upstream = _make_upstream()
        node = SqueezeNode()
        node._node_id = "sq1"
        result = node.execute(input=upstream, mode="mean")
        assert result[0].node.module_type == "Squeeze"
        assert result[0].node.params["mode"] == "mean"

    def test_execute_default_mode(self):
        upstream = _make_upstream()
        node = SqueezeNode()
        node._node_id = "sq1"
        result = node.execute(input=upstream)
        assert result[0].node.params["mode"] == "first"

    def test_on_disable(self):
        upstream = _make_upstream()
        node = SqueezeNode()
        node._node_id = "sq1"
        result = node.on_disable(input=upstream)
        assert result[0] is upstream


# ---------------------------------------------------------------------------
# Module builder tests
# ---------------------------------------------------------------------------

class TestModuleBuilders:
    def test_self_attention_module(self):
        mod = _SelfAttentionModule(32, 4)
        x = torch.randn(2, 8, 32)
        out = mod(x)
        assert out.shape == (2, 8, 32)

    def test_cross_attention_module(self):
        mod = _CrossAttentionModule(32, 4)
        query = torch.randn(2, 1, 32)
        context = torch.randn(2, 8, 32)
        out = mod(query, context)
        assert out.shape == (2, 1, 32)

    def test_positional_encoding_module(self):
        mod = _PositionalEncodingModule(256, 32)
        x = torch.zeros(2, 8, 32)
        out = mod(x)
        assert out.shape == (2, 8, 32)
        assert not torch.equal(out, x)  # PE was added

    def test_positional_encoding_shorter_than_max(self):
        mod = _PositionalEncodingModule(256, 32)
        x = torch.zeros(2, 4, 32)
        out = mod(x)
        assert out.shape == (2, 4, 32)

    def test_tokenize_op(self):
        from app.engine.graph_module import _build_tokenize_op
        op = _build_tokenize_op({"n_tokens": 64})
        x = torch.randn(2, 64)
        out = op(x)
        assert out.shape == (2, 64, 1)

    def test_tokenize_op_multi_dim(self):
        from app.engine.graph_module import _build_tokenize_op
        op = _build_tokenize_op({"n_tokens": 8})
        x = torch.randn(2, 32)
        out = op(x)
        assert out.shape == (2, 8, 4)

    def test_squeeze_op_first(self):
        from app.engine.graph_module import _build_squeeze_op
        op = _build_squeeze_op({"mode": "first"})
        x = torch.randn(2, 5, 32)
        out = op(x)
        assert out.shape == (2, 32)
        assert torch.equal(out, x[:, 0, :])

    def test_squeeze_op_mean(self):
        from app.engine.graph_module import _build_squeeze_op
        op = _build_squeeze_op({"mode": "mean"})
        x = torch.randn(2, 5, 32)
        out = op(x)
        assert out.shape == (2, 32)

    def test_squeeze_op_flatten(self):
        from app.engine.graph_module import _build_squeeze_op
        op = _build_squeeze_op({"mode": "flatten"})
        x = torch.randn(2, 4, 8)
        out = op(x)
        assert out.shape == (2, 32)


# ---------------------------------------------------------------------------
# Shape inference tests
# ---------------------------------------------------------------------------

class TestShapeInference:
    def test_tokenize_shape(self):
        nodes = trace_graph([
            ArchRef(ArchNode("l1", "Linear", {"in_features": 4, "out_features": 64}, []), 0)
        ])
        tok_node = ArchNode("tok", "Tokenize", {"n_tokens": 64}, [ArchRef(nodes[0], 0)])
        all_nodes = nodes + [tok_node]
        infer_shapes_graph(all_nodes, input_dim=4)
        # Can't directly access shapes, but we can test via downstream Linear

    def test_linear_after_tokenize(self):
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=64)[0]

        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=64)[0]

        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=32)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        l2_node = [n for n in nodes if n.node_id == "l2"][0]
        assert l2_node.params["in_features"] == 1  # 64 tokens of dim 1

    def test_self_attention_fills_embed_dim(self):
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=64)[0]

        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=8)[0]

        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=32)[0]

        sa = SelfAttentionNode(); sa._node_id = "sa"
        ref = sa.execute(input=ref, num_heads=4)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        sa_node = [n for n in nodes if n.node_id == "sa"][0]
        assert sa_node.params["embed_dim"] == 32

    def test_cross_attention_fills_embed_dim(self):
        # Query path
        l1 = LinearNode(); l1._node_id = "l1"
        ref_q = l1.execute(in_features=2, out_features=32)[0]
        tok_q = TokenizeNode(); tok_q._node_id = "tok_q"
        ref_q = tok_q.execute(input=ref_q, n_tokens=1)[0]

        # Context path
        l2 = LinearNode(); l2._node_id = "l2"
        ref_c = l2.execute(in_features=64, out_features=64)[0]
        tok_c = TokenizeNode(); tok_c._node_id = "tok_c"
        ref_c = tok_c.execute(input=ref_c, n_tokens=8)[0]
        l3 = LinearNode(); l3._node_id = "l3"
        ref_c = l3.execute(input=ref_c, out_features=32)[0]

        ca = CrossAttentionNode(); ca._node_id = "ca"
        ref = ca.execute(input_query=ref_q, input_context=ref_c, num_heads=4)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=None)
        ca_node = [n for n in nodes if n.node_id == "ca"][0]
        assert ca_node.params["embed_dim"] == 32

    def test_positional_encoding_fills_embed_dim(self):
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=32)[0]

        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=4)[0]

        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=16)[0]

        pe = PositionalEncodingNode(); pe._node_id = "pe"
        ref = pe.execute(input=ref, max_len=256)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        pe_node = [n for n in nodes if n.node_id == "pe"][0]
        assert pe_node.params["embed_dim"] == 16

    def test_squeeze_first_shape(self):
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=32)[0]

        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=1)[0]

        sq = SqueezeNode(); sq._node_id = "sq"
        ref = sq.execute(input=ref, mode="first")[0]

        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=1)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        l2_node = [n for n in nodes if n.node_id == "l2"][0]
        assert l2_node.params["in_features"] == 32

    def test_squeeze_flatten_shape(self):
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=32)[0]

        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=4)[0]

        sq = SqueezeNode(); sq._node_id = "sq"
        ref = sq.execute(input=ref, mode="flatten")[0]

        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=1)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        l2_node = [n for n in nodes if n.node_id == "l2"][0]
        assert l2_node.params["in_features"] == 32  # 4 tokens * 8 dim


# ---------------------------------------------------------------------------
# Integration tests: full forward pass through GraphModule
# ---------------------------------------------------------------------------

class TestIntegrationSelfAttention:
    def test_simple_self_attention_path(self):
        """Linear → Tokenize → Linear → SelfAttn → Squeeze → Linear."""
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=64)[0]

        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=8)[0]

        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=32)[0]

        sa = SelfAttentionNode(); sa._node_id = "sa"
        ref = sa.execute(input=ref, num_heads=4)[0]

        sq = SqueezeNode(); sq._node_id = "sq"
        ref = sq.execute(input=ref, mode="mean")[0]

        l3 = LinearNode(); l3._node_id = "l3"
        ref = l3.execute(input=ref, out_features=1)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [ref])

        x = torch.randn(5, 4)
        y = model(x)
        assert y.shape == (5, 1)


class TestIntegrationCrossAttention:
    def test_two_branch_cross_attention(self):
        """Split → [Tokenize → Linear → SelfAttn, Linear → Tokenize] → CrossAttn → Squeeze → Linear."""
        l0 = LinearNode(); l0._node_id = "l0"
        ref = l0.execute(in_features=10, out_features=10)[0]

        split = SplitNode(); split._node_id = "split"
        branch, trunk = split.execute(input=ref, split_sizes="8,2")

        # Branch: 8 sensors → tokens → embed → self-attn
        tok_b = TokenizeNode(); tok_b._node_id = "tok_b"
        ref_b = tok_b.execute(input=branch, n_tokens=8)[0]
        l_b = LinearNode(); l_b._node_id = "l_b"
        ref_b = l_b.execute(input=ref_b, out_features=16)[0]
        sa = SelfAttentionNode(); sa._node_id = "sa"
        ref_b = sa.execute(input=ref_b, num_heads=4)[0]

        # Trunk: 2 coords → embed → single token
        l_t = LinearNode(); l_t._node_id = "l_t"
        ref_t = l_t.execute(input=trunk, out_features=16)[0]
        tok_t = TokenizeNode(); tok_t._node_id = "tok_t"
        ref_t = tok_t.execute(input=ref_t, n_tokens=1)[0]

        # Cross-attention
        ca = CrossAttentionNode(); ca._node_id = "ca"
        ref = ca.execute(input_query=ref_t, input_context=ref_b, num_heads=4)[0]

        sq = SqueezeNode(); sq._node_id = "sq"
        ref = sq.execute(input=ref, mode="first")[0]

        l_out = LinearNode(); l_out._node_id = "l_out"
        ref = l_out.execute(input=ref, out_features=1)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=10)
        model = GraphModule(nodes, [ref])

        x = torch.randn(5, 10)
        y = model(x)
        assert y.shape == (5, 1)


class TestIntegrationAttentionONet:
    def test_full_attention_onet_topology(self):
        """Full AttentionONet: matches the saved graph structure."""
        # Input
        l_in = LinearNode(); l_in._node_id = "input_linear"
        ref = l_in.execute(in_features=66, out_features=66)[0]

        split = SplitNode(); split._node_id = "split"
        sensors, query = split.execute(input=ref, split_sizes="64,2")

        # Branch: sensors → tokenize → embed → PE → self-attn
        tok_s = TokenizeNode(); tok_s._node_id = "tokenize_sensors"
        ref_b = tok_s.execute(input=sensors, n_tokens=64)[0]
        l_embed = LinearNode(); l_embed._node_id = "sensor_embed"
        ref_b = l_embed.execute(input=ref_b, out_features=32)[0]
        pe = PositionalEncodingNode(); pe._node_id = "pos_enc"
        ref_b = pe.execute(input=ref_b, max_len=256)[0]
        sa = SelfAttentionNode(); sa._node_id = "self_attn"
        ref_b = sa.execute(input=ref_b, num_heads=4)[0]

        # Trunk: query → linear → tokenize
        l_trunk = LinearNode(); l_trunk._node_id = "trunk_linear"
        ref_t = l_trunk.execute(input=query, out_features=32)[0]
        tok_q = TokenizeNode(); tok_q._node_id = "tokenize_query"
        ref_t = tok_q.execute(input=ref_t, n_tokens=1)[0]

        # Cross-attention → squeeze → output
        ca = CrossAttentionNode(); ca._node_id = "cross_attn"
        ref = ca.execute(input_query=ref_t, input_context=ref_b, num_heads=4)[0]
        sq = SqueezeNode(); sq._node_id = "squeeze"
        ref = sq.execute(input=ref, mode="first")[0]
        l_out = LinearNode(); l_out._node_id = "output_linear"
        ref = l_out.execute(input=ref, out_features=1)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=66)
        model = GraphModule(nodes, [ref])

        x = torch.randn(8, 66)
        y = model(x)
        assert y.shape == (8, 1)

    def test_gradient_flows(self):
        """All attention params receive gradients."""
        l_in = LinearNode(); l_in._node_id = "l_in"
        ref = l_in.execute(in_features=8, out_features=8)[0]
        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=4)[0]
        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=16)[0]
        sa = SelfAttentionNode(); sa._node_id = "sa"
        ref = sa.execute(input=ref, num_heads=4)[0]
        sq = SqueezeNode(); sq._node_id = "sq"
        ref = sq.execute(input=ref, mode="mean")[0]
        l3 = LinearNode(); l3._node_id = "l3"
        ref = l3.execute(input=ref, out_features=1)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=8)
        model = GraphModule(nodes, [ref])

        x = torch.randn(4, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_num_heads(self):
        """Self-attention works with 1, 2, 4, 8 heads."""
        for heads in [1, 2, 4, 8]:
            l1 = LinearNode(); l1._node_id = "l1"
            ref = l1.execute(in_features=4, out_features=32)[0]
            tok = TokenizeNode(); tok._node_id = "tok"
            ref = tok.execute(input=ref, n_tokens=4)[0]
            l2 = LinearNode(); l2._node_id = "l2"
            ref = l2.execute(input=ref, out_features=8)[0]
            sa = SelfAttentionNode(); sa._node_id = "sa"
            ref = sa.execute(input=ref, num_heads=heads)[0]
            sq = SqueezeNode(); sq._node_id = "sq"
            ref = sq.execute(input=ref, mode="first")[0]

            nodes = trace_graph([ref])
            infer_shapes_graph(nodes, input_dim=4)
            model = GraphModule(nodes, [ref])
            y = model(torch.randn(2, 4))
            assert y.shape == (2, 8)

    def test_batch_size_one(self):
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=8)[0]
        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=4)[0]
        sa = SelfAttentionNode(); sa._node_id = "sa"
        ref = sa.execute(input=ref, num_heads=1)[0]
        sq = SqueezeNode(); sq._node_id = "sq"
        ref = sq.execute(input=ref, mode="mean")[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [ref])
        y = model(torch.randn(1, 4))
        assert y.shape == (1, 2)


class TestAblation:
    def test_disable_self_attention(self):
        """Disabled self-attention passes tokens through unchanged."""
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=32)[0]
        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=8)[0]
        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=16)[0]

        # Disabled self-attention → passthrough
        sa = SelfAttentionNode(); sa._node_id = "sa"
        ref = sa.on_disable(input=ref)[0]

        sq = SqueezeNode(); sq._node_id = "sq"
        ref = sq.execute(input=ref, mode="mean")[0]
        l3 = LinearNode(); l3._node_id = "l3"
        ref = l3.execute(input=ref, out_features=1)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [ref])
        y = model(torch.randn(5, 4))
        assert y.shape == (5, 1)

    def test_disable_cross_attention(self):
        """Disabled cross-attention returns query path only."""
        q = _make_upstream("q", 32, 2)
        c = _make_upstream("c", 32, 8)

        tok_q = TokenizeNode(); tok_q._node_id = "tok_q"
        ref_q = tok_q.execute(input=q, n_tokens=1)[0]

        tok_c = TokenizeNode(); tok_c._node_id = "tok_c"
        ref_c = tok_c.execute(input=c, n_tokens=8)[0]

        ca = CrossAttentionNode(); ca._node_id = "ca"
        ref = ca.on_disable(input_query=ref_q, input_context=ref_c)[0]
        # on_disable returns query input, which is ref_q (Tokenize output)
        assert ref is ref_q


class TestWithPositionalEncoding:
    def test_pe_in_full_path(self):
        """Linear → Tokenize → Linear → PE → SelfAttn → Squeeze → Linear."""
        l1 = LinearNode(); l1._node_id = "l1"
        ref = l1.execute(in_features=4, out_features=16)[0]
        tok = TokenizeNode(); tok._node_id = "tok"
        ref = tok.execute(input=ref, n_tokens=4)[0]
        l2 = LinearNode(); l2._node_id = "l2"
        ref = l2.execute(input=ref, out_features=16)[0]
        pe = PositionalEncodingNode(); pe._node_id = "pe"
        ref = pe.execute(input=ref, max_len=256)[0]
        sa = SelfAttentionNode(); sa._node_id = "sa"
        ref = sa.execute(input=ref, num_heads=4)[0]
        sq = SqueezeNode(); sq._node_id = "sq"
        ref = sq.execute(input=ref, mode="first")[0]
        l3 = LinearNode(); l3._node_id = "l3"
        ref = l3.execute(input=ref, out_features=1)[0]

        nodes = trace_graph([ref])
        infer_shapes_graph(nodes, input_dim=4)
        model = GraphModule(nodes, [ref])

        x = torch.randn(3, 4)
        y = model(x)
        assert y.shape == (3, 1)
