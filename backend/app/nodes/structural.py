"""Structural operation nodes: Split, Concat, DotProduct, Add.

These are DAG operations that route or combine tensors.
They have no trainable parameters — they're pure tensor operations.
"""
from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry
from ..engine.graph_module import ArchNode, ArchRef


@NodeRegistry.register("Split")
class SplitNode(BaseNode):
    CATEGORY = "Structural"
    DISPLAY_NAME = "Split"
    DESCRIPTION = "Split tensor along feature dimension into two parts"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "split_sizes": InputSpec(
                dtype=DataType.STRING, default="32,32", required=True,
                is_handle=False,
            ),
            "dim": InputSpec(
                dtype=DataType.INT, default=-1, required=False,
                is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [
            OutputSpec(dtype=DataType.ARCH, name="output_a"),
            OutputSpec(dtype=DataType.ARCH, name="output_b"),
        ]

    def execute(self, **kwargs) -> tuple:
        upstream = kwargs["input"]
        sizes = [int(s.strip()) for s in kwargs["split_sizes"].split(",")]
        dim = kwargs.get("dim", -1)
        node = ArchNode(
            node_id=self._node_id,
            module_type="Split",
            params={"split_sizes": sizes, "dim": dim},
            inputs=[upstream],
            n_outputs=2,
        )
        return (ArchRef(node, 0), ArchRef(node, 1))

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        return (upstream, upstream)


@NodeRegistry.register("Concat")
class ConcatNode(BaseNode):
    CATEGORY = "Structural"
    DISPLAY_NAME = "Concat"
    DESCRIPTION = "Concatenate two tensors along feature dimension"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input_a": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "input_b": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "dim": InputSpec(
                dtype=DataType.INT, default=-1, required=False,
                is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        a = kwargs["input_a"]
        b = kwargs["input_b"]
        dim = kwargs.get("dim", -1)
        node = ArchNode(
            node_id=self._node_id,
            module_type="Concat",
            params={"dim": dim},
            inputs=[a, b],
        )
        return (ArchRef(node, 0),)

    def on_disable(self, **kwargs) -> tuple:
        return (kwargs["input_a"],)


@NodeRegistry.register("DotProduct")
class DotProductNode(BaseNode):
    CATEGORY = "Structural"
    DISPLAY_NAME = "Dot Product"
    DESCRIPTION = "Element-wise multiply and sum: sum(a * b, dim=-1)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input_a": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "input_b": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        a = kwargs["input_a"]
        b = kwargs["input_b"]
        node = ArchNode(
            node_id=self._node_id,
            module_type="DotProduct",
            params={},
            inputs=[a, b],
        )
        return (ArchRef(node, 0),)

    def on_disable(self, **kwargs) -> tuple:
        return (kwargs["input_a"],)


@NodeRegistry.register("Add")
class AddNode(BaseNode):
    CATEGORY = "Structural"
    DISPLAY_NAME = "Add"
    DESCRIPTION = "Element-wise addition (for skip connections / residuals)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input_a": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
            "input_b": InputSpec(dtype=DataType.ARCH, required=True, is_handle=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        a = kwargs["input_a"]
        b = kwargs["input_b"]
        node = ArchNode(
            node_id=self._node_id,
            module_type="Add",
            params={},
            inputs=[a, b],
        )
        return (ArchRef(node, 0),)

    def on_disable(self, **kwargs) -> tuple:
        return (kwargs["input_a"],)
