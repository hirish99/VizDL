"""Model export node: save trained weights and full training report to disk."""
import json
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry
from ..config import settings


def _describe_architecture(model: nn.Module) -> str:
    """Generate a short architecture description like 'Linear32-ReLU-Linear16-ReLU-Linear1'."""
    parts = []
    for layer in model.children():
        name = type(layer).__name__
        if isinstance(layer, nn.Linear):
            parts.append(f"Linear{layer.out_features}")
        elif isinstance(layer, nn.Dropout):
            parts.append(f"Drop{layer.p}")
        elif isinstance(layer, nn.BatchNorm1d):
            parts.append(f"BN{layer.num_features}")
        else:
            parts.append(name)
    return "-".join(parts) if parts else "model"


@NodeRegistry.register("ModelExport")
class ModelExportNode(BaseNode):
    CATEGORY = "Model"
    DISPLAY_NAME = "Model Export"
    DESCRIPTION = "Save model weights + full training report (loss curves, test results, architecture)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "training_result": InputSpec(dtype=DataType.TRAINING_RESULT, required=True),
            "test_metrics": InputSpec(dtype=DataType.METRICS, required=False),
            "name": InputSpec(
                dtype=DataType.STRING, default="", required=False,
                is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.STRING, name="file_path")]

    def execute(self, **kwargs) -> tuple:
        result = kwargs["training_result"]
        test_metrics = kwargs.get("test_metrics")
        model: nn.Module = result["model"]
        custom_name = kwargs.get("name", "").strip()

        arch = _describe_architecture(model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if custom_name:
            base_name = f"{custom_name}_{timestamp}"
        else:
            base_name = f"{arch}_{timestamp}"

        # --- Save weights (.pt) ---
        weights_path = settings.weights_dir / f"{base_name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "architecture": arch,
            "model_structure": str(model),
            "saved_at": timestamp,
        }, weights_path)

        # --- Build comprehensive report ---
        history = result.get("history", {})
        report = {
            "model": {
                "architecture": arch,
                "structure": str(model),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "device": str(next(model.parameters()).device),
            },
            "training": {
                "epochs": history.get("epoch", []),
                "train_loss": history.get("train_loss", []),
                "val_loss": history.get("val_loss", []),
                "final_train_loss": result.get("final_train_loss"),
                "final_val_loss": result.get("final_val_loss"),
                "total_epochs": len(history.get("epoch", [])),
            },
            "timestamp": timestamp,
            "weights_file": f"{base_name}.pt",
        }

        # Include test results if Evaluator is connected
        if test_metrics is not None:
            report["test_results"] = {
                "mse": test_metrics.get("mse"),
                "mae": test_metrics.get("mae"),
                "r2": test_metrics.get("r2"),
                "n_samples": test_metrics.get("n_samples"),
                "predictions": test_metrics.get("predictions"),
                "targets": test_metrics.get("targets"),
            }

        # --- Save report (.json) ---
        report_path = settings.weights_dir / f"{base_name}_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))

        return (str(weights_path),)
