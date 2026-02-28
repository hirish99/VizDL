"""Optimizer nodes: SGD, Adam, AdamW."""
import torch.optim as optim

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


@NodeRegistry.register("SGD")
class SGDNode(BaseNode):
    CATEGORY = "Optimizer"
    DISPLAY_NAME = "SGD"
    DESCRIPTION = "Stochastic Gradient Descent optimizer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "model": InputSpec(dtype=DataType.MODEL, required=True),
            "lr": InputSpec(
                dtype=DataType.FLOAT, default=0.01, required=False,
                min_val=1e-8, max_val=10.0, is_handle=False,
            ),
            "momentum": InputSpec(
                dtype=DataType.FLOAT, default=0.0, required=False,
                min_val=0.0, max_val=1.0, is_handle=False,
            ),
            "weight_decay": InputSpec(
                dtype=DataType.FLOAT, default=0.0, required=False,
                min_val=0.0, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.OPTIMIZER, name="optimizer")]

    def execute(self, **kwargs) -> tuple:
        model = kwargs["model"]
        lr = kwargs.get("lr", 0.01)
        momentum = kwargs.get("momentum", 0.0)
        weight_decay = kwargs.get("weight_decay", 0.0)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        return (optimizer,)


@NodeRegistry.register("Adam")
class AdamNode(BaseNode):
    CATEGORY = "Optimizer"
    DISPLAY_NAME = "Adam"
    DESCRIPTION = "Adam optimizer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "model": InputSpec(dtype=DataType.MODEL, required=True),
            "lr": InputSpec(
                dtype=DataType.FLOAT, default=0.001, required=False,
                min_val=1e-8, max_val=10.0, is_handle=False,
            ),
            "weight_decay": InputSpec(
                dtype=DataType.FLOAT, default=0.0, required=False,
                min_val=0.0, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.OPTIMIZER, name="optimizer")]

    def execute(self, **kwargs) -> tuple:
        model = kwargs["model"]
        lr = kwargs.get("lr", 0.001)
        weight_decay = kwargs.get("weight_decay", 0.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return (optimizer,)


@NodeRegistry.register("AdamW")
class AdamWNode(BaseNode):
    CATEGORY = "Optimizer"
    DISPLAY_NAME = "AdamW"
    DESCRIPTION = "AdamW optimizer (decoupled weight decay)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "model": InputSpec(dtype=DataType.MODEL, required=True),
            "lr": InputSpec(
                dtype=DataType.FLOAT, default=0.001, required=False,
                min_val=1e-8, max_val=10.0, is_handle=False,
            ),
            "weight_decay": InputSpec(
                dtype=DataType.FLOAT, default=0.01, required=False,
                min_val=0.0, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.OPTIMIZER, name="optimizer")]

    def execute(self, **kwargs) -> tuple:
        model = kwargs["model"]
        lr = kwargs.get("lr", 0.001)
        weight_decay = kwargs.get("weight_decay", 0.01)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        return (optimizer,)
