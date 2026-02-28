"""Tests documenting non-layer node on_disable behavior.

These tests document the current (broken) behavior of disabling non-layer nodes.
They use xfail markers to flag known issues that should be fixed when the
semantics of "disabling a loss/optimizer/etc." are decided.
"""
import pytest
import torch
import torch.nn as nn

from app.nodes.data import CSVLoaderNode, DataSplitterNode
from app.nodes.losses import MSELossNode, CrossEntropyLossNode, L1LossNode
from app.nodes.optimizers import SGDNode, AdamNode, AdamWNode
from app.nodes.training import TrainingLoopNode
from app.nodes.metrics import MetricsCollectorNode
from app.nodes.model_assembly import GraphModelNode


class TestDataSplitterAblation:
    def test_on_disable_returns_wrong_output_count(self):
        """DataSplitter.on_disable returns 1-tuple, but pipeline expects 2 outputs.

        The pipeline destructures: train_loader, val_loader = splitter.execute(...)
        With on_disable, it returns (dataset,) which cannot be unpacked into 2 values.
        """
        node = DataSplitterNode()
        dataset = {"X": torch.randn(10, 4), "y": torch.randn(10, 1)}
        result = node.on_disable(dataset=dataset)
        # Default on_disable returns first input as 1-tuple
        assert len(result) == 1
        # This is wrong — DataSplitter has 2 outputs (train_loader, val_loader)
        assert len(DataSplitterNode.RETURN_TYPES()) == 2


class TestLossNodeAblation:
    def test_mse_loss_on_disable_returns_none(self):
        """Loss nodes have no inputs, so on_disable returns (None,)."""
        node = MSELossNode()
        result = node.on_disable()
        assert result == (None,)

    def test_cross_entropy_on_disable_returns_none(self):
        node = CrossEntropyLossNode()
        result = node.on_disable()
        assert result == (None,)

    def test_l1_loss_on_disable_returns_none(self):
        node = L1LossNode()
        result = node.on_disable()
        assert result == (None,)

    @pytest.mark.xfail(reason="Disabled loss node passes None to training loop, causing crash")
    def test_none_loss_fn_breaks_training(self):
        """Passing None as loss_fn to TrainingLoop would crash."""
        loss_fn = MSELossNode().on_disable()[0]
        assert loss_fn is not None  # This fails — documents the bug


class TestOptimizerAblation:
    def test_adam_on_disable_returns_model(self):
        """Optimizer.on_disable returns the model (first input), not an optimizer."""
        model = nn.Sequential(nn.Linear(4, 1))
        node = AdamNode()
        result = node.on_disable(model=model, lr=0.001)
        assert result == (model,)
        assert isinstance(result[0], nn.Module)

    def test_sgd_on_disable_returns_model(self):
        model = nn.Sequential(nn.Linear(4, 1))
        node = SGDNode()
        result = node.on_disable(model=model, lr=0.01)
        assert result == (model,)

    def test_adamw_on_disable_returns_model(self):
        model = nn.Sequential(nn.Linear(4, 1))
        node = AdamWNode()
        result = node.on_disable(model=model, lr=0.001)
        assert result == (model,)

    @pytest.mark.xfail(reason="Disabled optimizer returns model instead of optimizer, crashes training")
    def test_model_as_optimizer_breaks_training(self):
        """TrainingLoop calls optimizer.zero_grad() which fails on nn.Module."""
        model = nn.Sequential(nn.Linear(4, 1))
        fake_optimizer = AdamNode().on_disable(model=model, lr=0.001)[0]
        # This would fail because nn.Module has no zero_grad() that works like an optimizer
        assert hasattr(fake_optimizer, "step") and callable(fake_optimizer.step)
        # nn.Module does not have optimizer.step()


class TestTrainingLoopAblation:
    def test_on_disable_returns_model(self):
        """TrainingLoop.on_disable returns the model (first kwarg), not a training result dict."""
        model = nn.Sequential(nn.Linear(4, 1))
        node = TrainingLoopNode()
        result = node.on_disable(model=model)
        assert result == (model,)

    @pytest.mark.xfail(reason="Disabled training loop returns model, not result dict — breaks metrics")
    def test_model_as_training_result_breaks_metrics(self):
        """MetricsCollector expects dict with 'history' key, not nn.Module."""
        model = nn.Sequential(nn.Linear(4, 1))
        fake_result = TrainingLoopNode().on_disable(model=model)[0]
        # MetricsCollector would try: training_result["history"]
        assert isinstance(fake_result, dict) and "history" in fake_result


class TestGraphModelAblation:
    def test_on_disable_passes_through_architecture(self):
        """GraphModel.on_disable returns architecture (first input), not a model."""
        from app.engine.graph_module import ArchNode, ArchRef
        arch_ref = ArchRef(ArchNode("l1", "Linear", {"in_features": 4, "out_features": 1}, []))
        node = GraphModelNode()
        node._node_id = "gm"
        result = node.on_disable(architecture=arch_ref)
        assert result == (arch_ref,)
        # This is an ArchRef, not an nn.Module — downstream nodes expecting a model will fail


class TestMetricsCollectorAblation:
    def test_on_disable_passes_through(self):
        node = MetricsCollectorNode()
        training_result = {"model": None, "history": {"train_loss": [1.0]}}
        result = node.on_disable(training_result=training_result)
        assert result == (training_result,)
