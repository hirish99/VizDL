"""Layer nodes: Linear, ReLU, GELU, Sigmoid, Tanh, Dropout, BatchNorm1d, LayerNorm.

Each layer has an optional `input` handle (ARCH type from previous layer)
and an `output` handle (ARCH type to next layer or GraphModel).
This lets you visually chain: Linear → ReLU → Linear → GraphModel,
or build DAGs with skip connections and parallel branches.
"""
from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry
from ..engine.graph_module import ArchNode, ArchRef


def _make_arch(node_id: str, module_type: str, params: dict, upstream: ArchRef | None) -> ArchRef:
    """Create an ArchNode and return an ArchRef to it."""
    inputs = [upstream] if upstream else []
    return ArchRef(ArchNode(node_id, module_type, params, inputs))


@NodeRegistry.register("Linear")
class LinearNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Linear"
    DESCRIPTION = "Fully connected linear layer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(
                dtype=DataType.ARCH, required=False, is_handle=True,
            ),
            "in_features": InputSpec(
                dtype=DataType.INT, default=None, required=False,
                min_val=1, is_handle=False,
            ),
            "out_features": InputSpec(
                dtype=DataType.INT, default=64, required=True,
                min_val=1, is_handle=False,
            ),
            "bias": InputSpec(
                dtype=DataType.BOOL, default=True, required=False,
                is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        return (_make_arch(self._node_id, "Linear", {
            "in_features": kwargs.get("in_features"),
            "out_features": kwargs["out_features"],
            "bias": kwargs.get("bias", True),
        }, kwargs.get("input")),)

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        return (_make_arch(self._node_id, "Identity", {}, upstream),)


@NodeRegistry.register("ReLU")
class ReLUNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "ReLU"
    DESCRIPTION = "ReLU activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=False, is_handle=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        return (_make_arch(self._node_id, "ReLU", {}, kwargs.get("input")),)

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        if upstream is not None:
            return (upstream,)
        return (_make_arch(self._node_id, "Identity", {}, None),)


@NodeRegistry.register("GELU")
class GELUNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "GELU"
    DESCRIPTION = "GELU activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=False, is_handle=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        return (_make_arch(self._node_id, "GELU", {}, kwargs.get("input")),)

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        if upstream is not None:
            return (upstream,)
        return (_make_arch(self._node_id, "Identity", {}, None),)


@NodeRegistry.register("Sigmoid")
class SigmoidNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Sigmoid"
    DESCRIPTION = "Sigmoid activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=False, is_handle=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        return (_make_arch(self._node_id, "Sigmoid", {}, kwargs.get("input")),)

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        if upstream is not None:
            return (upstream,)
        return (_make_arch(self._node_id, "Identity", {}, None),)


@NodeRegistry.register("Tanh")
class TanhNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Tanh"
    DESCRIPTION = "Tanh activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=False, is_handle=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        return (_make_arch(self._node_id, "Tanh", {}, kwargs.get("input")),)

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        if upstream is not None:
            return (upstream,)
        return (_make_arch(self._node_id, "Identity", {}, None),)


@NodeRegistry.register("Dropout")
class DropoutNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Dropout"
    DESCRIPTION = "Dropout regularization"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=False, is_handle=True),
            "p": InputSpec(
                dtype=DataType.FLOAT, default=0.5, required=False,
                min_val=0.0, max_val=1.0, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        return (_make_arch(self._node_id, "Dropout", {
            "p": kwargs.get("p", 0.5),
        }, kwargs.get("input")),)

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        if upstream is not None:
            return (upstream,)
        return (_make_arch(self._node_id, "Identity", {}, None),)


@NodeRegistry.register("BatchNorm1d")
class BatchNorm1dNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "BatchNorm1d"
    DESCRIPTION = "1D batch normalization"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=False, is_handle=True),
            "num_features": InputSpec(
                dtype=DataType.INT, default=None, required=False,
                min_val=1, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        return (_make_arch(self._node_id, "BatchNorm1d", {
            "num_features": kwargs.get("num_features"),
        }, kwargs.get("input")),)

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        return (_make_arch(self._node_id, "Identity", {}, upstream),)


@NodeRegistry.register("LayerNorm")
class LayerNormNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "LayerNorm"
    DESCRIPTION = "Layer normalization"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "input": InputSpec(dtype=DataType.ARCH, required=False, is_handle=True),
            "normalized_shape": InputSpec(
                dtype=DataType.INT, default=None, required=False,
                min_val=1, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs) -> tuple:
        return (_make_arch(self._node_id, "LayerNorm", {
            "normalized_shape": kwargs.get("normalized_shape"),
        }, kwargs.get("input")),)

    def on_disable(self, **kwargs) -> tuple:
        upstream = kwargs.get("input")
        return (_make_arch(self._node_id, "Identity", {}, upstream),)
