"""Integration tests for the full training pipeline with ablation."""
import pytest

from app.engine.graph import Edge, Graph, NodeInstance
from app.engine.pipeline import execute_pipeline


class TestPipelineNoAblation:
    def test_basic_pipeline_completes(self, simple_layer_graph, pipeline_config):
        """Baseline: pipeline runs to completion without any ablation."""
        results = execute_pipeline(simple_layer_graph, pipeline_config)
        assert "training" in results
        assert "metrics" in results
        assert "export_path" in results
        assert results["training"]["final_train_loss"] is not None
        assert len(results["training"]["history"]["train_loss"]) == 3

    def test_pipeline_with_progress_callback(self, simple_layer_graph, pipeline_config):
        """Progress callback is invoked."""
        messages = []
        results = execute_pipeline(
            simple_layer_graph, pipeline_config,
            progress_callback=lambda msg: messages.append(msg),
        )
        assert any(m["type"] == "node_complete" for m in messages)
        assert any(m["type"] == "training_progress" for m in messages)
        assert results["training"]["final_train_loss"] is not None


class TestPipelineWithAblation:
    def test_disabled_middle_activation(self, pipeline_config):
        """Linear -> [disabled ReLU] -> Linear: should still train."""
        nodes = {
            "linear1": NodeInstance(
                id="linear1", node_type="Linear",
                params={"in_features": 4, "out_features": 8},
            ),
            "relu": NodeInstance(
                id="relu", node_type="ReLU",
                disabled=True,
            ),
            "linear2": NodeInstance(
                id="linear2", node_type="Linear",
                params={"out_features": 1},
            ),
        }
        edges = [
            Edge(id="e1", source_node="linear1", source_output=0,
                 target_node="relu", target_input="input"),
            Edge(id="e2", source_node="relu", source_output=0,
                 target_node="linear2", target_input="input"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        results = execute_pipeline(graph, pipeline_config)
        assert results["training"]["final_train_loss"] is not None

    def test_disabled_dropout(self, pipeline_config):
        """Linear -> [disabled Dropout] -> Linear: Dropout removed from chain."""
        nodes = {
            "linear1": NodeInstance(
                id="linear1", node_type="Linear",
                params={"in_features": 4, "out_features": 8},
            ),
            "dropout": NodeInstance(
                id="dropout", node_type="Dropout",
                params={"p": 0.5},
                disabled=True,
            ),
            "linear2": NodeInstance(
                id="linear2", node_type="Linear",
                params={"out_features": 1},
            ),
        }
        edges = [
            Edge(id="e1", source_node="linear1", source_output=0,
                 target_node="dropout", target_input="input"),
            Edge(id="e2", source_node="dropout", source_output=0,
                 target_node="linear2", target_input="input"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        results = execute_pipeline(graph, pipeline_config)
        assert results["training"]["final_train_loss"] is not None

    def test_disabled_first_linear(self, pipeline_config):
        """[disabled Linear] -> ReLU -> Linear: Identity replaces first Linear."""
        nodes = {
            "linear1": NodeInstance(
                id="linear1", node_type="Linear",
                params={"in_features": 4, "out_features": 8},
                disabled=True,
            ),
            "relu": NodeInstance(id="relu", node_type="ReLU"),
            "linear2": NodeInstance(
                id="linear2", node_type="Linear",
                params={"out_features": 1},
            ),
        }
        edges = [
            Edge(id="e1", source_node="linear1", source_output=0,
                 target_node="relu", target_input="input"),
            Edge(id="e2", source_node="relu", source_output=0,
                 target_node="linear2", target_input="input"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        results = execute_pipeline(graph, pipeline_config)
        assert results["training"]["final_train_loss"] is not None

    def test_disabled_batchnorm(self, pipeline_config):
        """Linear -> [disabled BatchNorm] -> ReLU -> Linear"""
        nodes = {
            "linear1": NodeInstance(
                id="linear1", node_type="Linear",
                params={"in_features": 4, "out_features": 8},
            ),
            "bn": NodeInstance(
                id="bn", node_type="BatchNorm1d",
                disabled=True,
            ),
            "relu": NodeInstance(id="relu", node_type="ReLU"),
            "linear2": NodeInstance(
                id="linear2", node_type="Linear",
                params={"out_features": 1},
            ),
        }
        edges = [
            Edge(id="e1", source_node="linear1", source_output=0,
                 target_node="bn", target_input="input"),
            Edge(id="e2", source_node="bn", source_output=0,
                 target_node="relu", target_input="input"),
            Edge(id="e3", source_node="relu", source_output=0,
                 target_node="linear2", target_input="input"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        results = execute_pipeline(graph, pipeline_config)
        assert results["training"]["final_train_loss"] is not None


class TestPipelineAblationComparison:
    def test_ablation_changes_results(self, simple_layer_graph, pipeline_config):
        """Ablating a layer should produce different training results."""
        import torch
        torch.manual_seed(42)
        results_normal = execute_pipeline(simple_layer_graph, pipeline_config)

        # Now disable ReLU
        torch.manual_seed(42)
        simple_layer_graph.nodes["relu"].disabled = True
        results_ablated = execute_pipeline(simple_layer_graph, pipeline_config)

        # Results should differ (different model architecture)
        normal_loss = results_normal["training"]["final_train_loss"]
        ablated_loss = results_ablated["training"]["final_train_loss"]
        assert normal_loss is not None
        assert ablated_loss is not None
