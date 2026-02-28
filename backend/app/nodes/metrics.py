"""Metrics collection node."""
from typing import Any

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


@NodeRegistry.register("MetricsCollector")
class MetricsCollectorNode(BaseNode):
    CATEGORY = "Metrics"
    DISPLAY_NAME = "Metrics Collector"
    DESCRIPTION = "Collects and formats training metrics"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "training_result": InputSpec(dtype=DataType.TRAINING_RESULT, required=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.METRICS, name="metrics")]

    def execute(self, **kwargs) -> tuple:
        result = kwargs["training_result"]
        history = result["history"]

        metrics = {
            "epochs": history["epoch"],
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "final_train_loss": result.get("final_train_loss"),
            "final_val_loss": result.get("final_val_loss"),
            "total_epochs": len(history["epoch"]),
        }

        return (metrics,)
