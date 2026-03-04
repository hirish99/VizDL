"""Loss function nodes."""
import torch
import torch.nn as nn

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


class RelativeMSELoss(nn.Module):
    """Relative L2 loss: sum(||pred - target||^2) / sum(||target||^2)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum((pred - target) ** 2) / (torch.sum(target ** 2) + 1e-8)


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


@NodeRegistry.register("RelativeMSELoss")
class RelativeMSELossNode(BaseNode):
    CATEGORY = "Loss"
    DISPLAY_NAME = "Relative MSE Loss"
    DESCRIPTION = "Relative L2 loss: ||pred - target||² / ||target||², standard for operator learning"

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LOSS_FN, name="loss_fn")]

    def execute(self, **kwargs) -> tuple:
        return (RelativeMSELoss(),)
