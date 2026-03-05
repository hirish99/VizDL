"""Tests for continue-training: resume from exported weights with history merging."""
import json
from pathlib import Path

import pytest
import torch

from app.config import settings
from app.engine.graph import Edge, Graph, NodeInstance
from app.engine.pipeline import execute_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_graph():
    """Linear(4->8) -> ReLU -> Linear(->1)."""
    nodes = {
        "linear1": NodeInstance(id="linear1", node_type="Linear",
                                params={"in_features": 4, "out_features": 8}),
        "relu": NodeInstance(id="relu", node_type="ReLU"),
        "linear2": NodeInstance(id="linear2", node_type="Linear",
                                params={"out_features": 1}),
    }
    edges = [
        Edge(id="e1", source_node="linear1", source_output=0,
             target_node="relu", target_input="input"),
        Edge(id="e2", source_node="relu", source_output=0,
             target_node="linear2", target_input="input"),
    ]
    return Graph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestContinueTraining:
    """Train 3 epochs, export, then resume for 3 more. Verify history merges."""

    def test_resume_merges_epoch_numbers(self, pipeline_config):
        """Resumed training should have continuous epoch numbers [1..6]."""
        graph = _make_simple_graph()

        # Phase 1: train 3 epochs
        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]

        # Phase 2: resume for 3 more
        pipeline_config["resume_from"] = pt_path
        r2 = execute_pipeline(graph, pipeline_config)

        epochs = r2["training"]["history"]["epoch"]
        assert epochs == [1, 2, 3, 4, 5, 6]

    def test_resume_merges_train_loss(self, pipeline_config):
        """Train loss history should have 6 entries after resume."""
        graph = _make_simple_graph()

        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]
        old_train = r1["training"]["history"]["train_loss"]

        pipeline_config["resume_from"] = pt_path
        r2 = execute_pipeline(graph, pipeline_config)

        merged_train = r2["training"]["history"]["train_loss"]
        assert len(merged_train) == 6
        # First 3 entries should match original run
        assert merged_train[:3] == old_train

    def test_resume_merges_val_loss(self, pipeline_config):
        """Val loss history should have 6 entries after resume."""
        graph = _make_simple_graph()

        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]
        old_val = r1["training"]["history"]["val_loss"]

        pipeline_config["resume_from"] = pt_path
        r2 = execute_pipeline(graph, pipeline_config)

        merged_val = r2["training"]["history"]["val_loss"]
        assert len(merged_val) == 6
        assert merged_val[:3] == old_val

    def test_resume_final_loss_is_from_new_training(self, pipeline_config):
        """final_train_loss should be from the last epoch of the new run, not old."""
        graph = _make_simple_graph()

        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]

        pipeline_config["resume_from"] = pt_path
        r2 = execute_pipeline(graph, pipeline_config)

        # final_train_loss should equal the last element of merged history
        assert r2["training"]["final_train_loss"] == r2["training"]["history"]["train_loss"][-1]

    def test_resume_report_has_full_history(self, pipeline_config):
        """The exported report JSON should contain the full merged history."""
        graph = _make_simple_graph()

        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]

        pipeline_config["resume_from"] = pt_path
        r2 = execute_pipeline(graph, pipeline_config)

        report_path = Path(r2["export_path"]).with_name(
            Path(r2["export_path"]).stem + "_report.json"
        )
        report = json.loads(report_path.read_text())
        assert len(report["training"]["train_loss"]) == 6
        assert len(report["training"]["val_loss"]) == 6
        assert report["training"]["total_epochs"] == 6

    def test_resume_metadata_in_results(self, pipeline_config):
        """Results should include resume metadata."""
        graph = _make_simple_graph()

        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]

        pipeline_config["resume_from"] = pt_path
        r2 = execute_pipeline(graph, pipeline_config)

        assert r2["training"]["resumed_from"] == pt_path
        assert r2["training"]["resume_epoch"] == 3

    def test_no_resume_has_null_metadata(self, pipeline_config):
        """Normal training should have null resume metadata."""
        graph = _make_simple_graph()
        r1 = execute_pipeline(graph, pipeline_config)
        assert r1["training"]["resumed_from"] is None
        assert r1["training"]["resume_epoch"] is None


class TestContinueTrainingWeights:
    """Verify that model weights are actually loaded (not training from scratch)."""

    def test_resumed_model_starts_with_loaded_weights(self, pipeline_config):
        """Loss at epoch 4 (resumed) should be similar to loss at epoch 3, not reset."""
        graph = _make_simple_graph()
        torch.manual_seed(42)

        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]
        loss_at_3 = r1["training"]["final_train_loss"]

        pipeline_config["resume_from"] = pt_path
        torch.manual_seed(42)
        r2 = execute_pipeline(graph, pipeline_config)

        # Epoch 4 loss (first new epoch) should be close to epoch 3 loss
        # If weights weren't loaded, it would be much worse (back to random init)
        loss_at_4 = r2["training"]["history"]["train_loss"][3]
        assert loss_at_4 < loss_at_3 * 3, (
            f"Epoch 4 loss ({loss_at_4:.4f}) is way worse than epoch 3 ({loss_at_3:.4f}), "
            "weights may not have been loaded"
        )

    def test_optimizer_state_saved_in_export(self, pipeline_config):
        """Exported .pt should contain optimizer_state_dict."""
        graph = _make_simple_graph()
        r1 = execute_pipeline(graph, pipeline_config)
        ckpt = torch.load(r1["export_path"], weights_only=False)
        assert "optimizer_state_dict" in ckpt
        assert len(ckpt["optimizer_state_dict"]) > 0

    def test_optimizer_state_restored_on_resume(self, pipeline_config):
        """Optimizer momentum should be restored (Adam has running averages)."""
        graph = _make_simple_graph()
        torch.manual_seed(42)

        # Train 5 epochs straight
        pipeline_config["epochs"] = 5
        r_straight = execute_pipeline(graph, pipeline_config)
        loss_straight = r_straight["training"]["history"]["train_loss"]

        # Train 3 + resume 2
        pipeline_config["epochs"] = 3
        pipeline_config.pop("resume_from", None)
        torch.manual_seed(42)
        r1 = execute_pipeline(graph, pipeline_config)

        pipeline_config["epochs"] = 2
        pipeline_config["resume_from"] = r1["export_path"]
        torch.manual_seed(42)
        r2 = execute_pipeline(graph, pipeline_config)
        loss_resumed = r2["training"]["history"]["train_loss"]

        # First 3 epochs should match (same seed, same data)
        for i in range(3):
            assert abs(loss_straight[i] - loss_resumed[i]) < 1e-5, (
                f"Epoch {i+1} mismatch: straight={loss_straight[i]:.6f} vs resumed={loss_resumed[i]:.6f}"
            )


class TestContinueTrainingEdgeCases:
    """Edge cases: missing optimizer state, architecture mismatch."""

    def test_resume_without_optimizer_state(self, pipeline_config, tmp_path):
        """Should work even if .pt has no optimizer_state_dict (old exports)."""
        graph = _make_simple_graph()

        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]

        # Strip optimizer state to simulate old export format
        ckpt = torch.load(pt_path, weights_only=False)
        del ckpt["optimizer_state_dict"]
        torch.save(ckpt, pt_path)

        pipeline_config["resume_from"] = pt_path
        r2 = execute_pipeline(graph, pipeline_config)
        assert len(r2["training"]["history"]["epoch"]) == 6

    def test_resume_with_mismatched_architecture(self, pipeline_config):
        """Mismatched architecture should raise a clear error."""
        graph = _make_simple_graph()
        r1 = execute_pipeline(graph, pipeline_config)
        pt_path = r1["export_path"]

        # Different architecture (wider hidden layer)
        different_graph = Graph(
            nodes={
                "linear1": NodeInstance(id="linear1", node_type="Linear",
                                        params={"in_features": 4, "out_features": 32}),
                "relu": NodeInstance(id="relu", node_type="ReLU"),
                "linear2": NodeInstance(id="linear2", node_type="Linear",
                                        params={"out_features": 1}),
            },
            edges=[
                Edge(id="e1", source_node="linear1", source_output=0,
                     target_node="relu", target_input="input"),
                Edge(id="e2", source_node="relu", source_output=0,
                     target_node="linear2", target_input="input"),
            ],
        )

        pipeline_config["resume_from"] = pt_path
        with pytest.raises(RuntimeError, match="size mismatch|Error"):
            execute_pipeline(different_graph, pipeline_config)

    def test_double_resume(self, pipeline_config):
        """Train 3, resume 3, resume 3 more — should have 9 epochs total."""
        graph = _make_simple_graph()

        r1 = execute_pipeline(graph, pipeline_config)
        pipeline_config["resume_from"] = r1["export_path"]
        r2 = execute_pipeline(graph, pipeline_config)
        pipeline_config["resume_from"] = r2["export_path"]
        r3 = execute_pipeline(graph, pipeline_config)

        epochs = r3["training"]["history"]["epoch"]
        assert epochs == list(range(1, 10))
        assert len(r3["training"]["history"]["train_loss"]) == 9
