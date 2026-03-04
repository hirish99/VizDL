"""Attention nodes: Tokenize, PositionalEncoding, SelfAttention, CrossAttention, Squeeze.

These enable attention-based architectures (e.g., AttentionONet) by introducing
3D (batch, seq, dim) tensor operations into the graph.
"""
from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry
from ..engine.graph_module import ArchNode, ArchRef


def _make_arch(node_id: str, module_type: str, params: dict, upstream: ArchRef | None) -> ArchRef:
    inputs = [upstream] if upstream else []
    return ArchRef(ArchNode(node_id, module_type, params, inputs))


@NodeRegistry.register("Tokenize")
class TokenizeNode(BaseNode):
    CATEGORY = "Structural"
    DISPLAY_NAME = "Tokenize"
    DESCRIPTION = "Reshape flat tensor into a sequence of tokens: (batch, N) → (batch, n_tokens, N/n_tokens)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "n_tokens": InputSpec(
                dtype=DataType.INT, default=64, required=True,
                min_val=1, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        n_tokens = kwargs["n_tokens"]
        return (_make_arch(self._node_id, "Tokenize", {"n_tokens": n_tokens}, upstream),)

    def on_disable(self, **kwargs) -> tuple:
        return (kwargs.get("input"),)


@NodeRegistry.register("PositionalEncoding")
class PositionalEncodingNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Positional Encoding"
    DESCRIPTION = "Add sinusoidal positional encoding to token embeddings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "max_len": InputSpec(
                dtype=DataType.INT, default=256, required=False,
                min_val=1, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        max_len = kwargs.get("max_len", 256)
        return (_make_arch(self._node_id, "PositionalEncoding", {"max_len": max_len}, upstream),)

    def on_disable(self, **kwargs) -> tuple:
        return (kwargs.get("input"),)


@NodeRegistry.register("SelfAttention")
class SelfAttentionNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Self-Attention"
    DESCRIPTION = "Multi-head self-attention (Q=K=V=input)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "num_heads": InputSpec(
                dtype=DataType.INT, default=4, required=True,
                min_val=1, is_handle=False,
            ),
            "dropout": InputSpec(
                dtype=DataType.FLOAT, default=0.0, required=False,
                min_val=0.0, max_val=1.0, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        num_heads = kwargs["num_heads"]
        dropout = kwargs.get("dropout", 0.0)
        return (_make_arch(self._node_id, "SelfAttention", {
            "num_heads": num_heads,
            "dropout": dropout,
        }, upstream),)

    def on_disable(self, **kwargs) -> tuple:
        return (kwargs.get("input"),)


@NodeRegistry.register("CrossAttention")
class CrossAttentionNode(BaseNode):
    CATEGORY = "Structural"
    DISPLAY_NAME = "Cross-Attention"
    DESCRIPTION = "Multi-head cross-attention (Q=query, K=V=context)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input_query": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "input_context": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "num_heads": InputSpec(
                dtype=DataType.INT, default=4, required=True,
                min_val=1, is_handle=False,
            ),
            "dropout": InputSpec(
                dtype=DataType.FLOAT, default=0.0, required=False,
                min_val=0.0, max_val=1.0, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        query = kwargs.get("input_query")
        context = kwargs.get("input_context")
        num_heads = kwargs["num_heads"]
        dropout = kwargs.get("dropout", 0.0)
        inputs = [x for x in [query, context] if x is not None]
        node = ArchNode(
            node_id=self._node_id,
            module_type="CrossAttention",
            params={"num_heads": num_heads, "dropout": dropout},
            inputs=inputs,
        )
        return (ArchRef(node, 0),)

    def on_disable(self, **kwargs) -> tuple:
        return (kwargs.get("input_query"),)


@NodeRegistry.register("Squeeze")
class SqueezeNode(BaseNode):
    CATEGORY = "Structural"
    DISPLAY_NAME = "Squeeze"
    DESCRIPTION = "Collapse sequence dimension: first token, mean pool, or flatten"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "mode": InputSpec(
                dtype=DataType.STRING, default="first", required=False,
                is_handle=False, choices=["first", "mean", "flatten"],
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        mode = kwargs.get("mode", "first")
        return (_make_arch(self._node_id, "Squeeze", {"mode": mode}, upstream),)

    def on_disable(self, **kwargs) -> tuple:
        return (kwargs.get("input"),)
