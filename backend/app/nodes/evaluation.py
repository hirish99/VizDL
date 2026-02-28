"""Evaluation node: run inference on test data and compare to ground truth."""
from typing import Any

import torch
import torch.nn as nn

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


@NodeRegistry.register("Evaluator")
class EvaluatorNode(BaseNode):
    CATEGORY = "Metrics"
    DISPLAY_NAME = "Evaluator"
    DESCRIPTION = "Evaluate a trained model on a test dataset, comparing predictions to ground truth"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "training_result": InputSpec(dtype=DataType.TRAINING_RESULT, required=True),
            "test_dataset": InputSpec(dtype=DataType.DATASET, required=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.METRICS, name="test_metrics")]

    def execute(self, **kwargs) -> tuple:
        result = kwargs["training_result"]
        test_data = kwargs["test_dataset"]

        model: nn.Module = result["model"]
        device = next(model.parameters()).device

        X_test = test_data["X"].to(device)
        y_test = test_data["y"].to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(X_test)

        # MSE
        mse = torch.mean((predictions - y_test) ** 2).item()
        # MAE
        mae = torch.mean(torch.abs(predictions - y_test)).item()
        # R²
        ss_res = torch.sum((y_test - predictions) ** 2).item()
        ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Per-sample predictions for inspection
        preds_list = predictions.cpu().squeeze().tolist()
        targets_list = y_test.cpu().squeeze().tolist()
        if not isinstance(preds_list, list):
            preds_list = [preds_list]
            targets_list = [targets_list]

        return ({
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "n_samples": len(X_test),
            "predictions": preds_list,
            "targets": targets_list,
        },)
