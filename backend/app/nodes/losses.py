"""Loss function nodes."""
import torch.nn as nn

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


@NodeRegistry.register("MSELoss")
class MSELossNode(BaseNode):
    CATEGORY = "Loss"
    DISPLAY_NAME = "MSE Loss"
    DESCRIPTION = "Mean Squared Error loss"

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LOSS_FN, name="loss_fn")]

    def execute(self, **kwargs) -> tuple:
        return (nn.MSELoss(),)


@NodeRegistry.register("CrossEntropyLoss")
class CrossEntropyLossNode(BaseNode):
    CATEGORY = "Loss"
    DISPLAY_NAME = "Cross Entropy Loss"
    DESCRIPTION = "Cross entropy loss for classification"

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LOSS_FN, name="loss_fn")]

    def execute(self, **kwargs) -> tuple:
        return (nn.CrossEntropyLoss(),)


@NodeRegistry.register("L1Loss")
class L1LossNode(BaseNode):
    CATEGORY = "Loss"
    DISPLAY_NAME = "L1 Loss"
    DESCRIPTION = "Mean Absolute Error loss"

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LOSS_FN, name="loss_fn")]

    def execute(self, **kwargs) -> tuple:
        return (nn.L1Loss(),)
