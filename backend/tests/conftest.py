"""Shared test fixtures for VisDL backend tests."""
import csv
import sys
from pathlib import Path

import pytest
import torch

# Ensure app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.engine.graph import Edge, Graph, NodeInstance


@pytest.fixture(scope="session", autouse=True)
def register_nodes():
    """Discover and register all node types once per test session."""
    from app.nodes.registry import NodeRegistry
    NodeRegistry.discover("app.nodes")


@pytest.fixture
def simple_layer_graph():
    """Linear(4->8) -> ReLU -> Linear(->1) graph."""
    nodes = {
        "linear1": NodeInstance(
            id="linear1", node_type="Linear",
            params={"in_features": 4, "out_features": 8},
        ),
        "relu": NodeInstance(
            id="relu", node_type="ReLU",
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
    return Graph(nodes=nodes, edges=edges)


@pytest.fixture
def sample_dataset():
    """Synthetic dataset dict with 100 samples, 4 features, 1 target."""
    torch.manual_seed(42)
    return {
        "X": torch.randn(100, 4),
        "y": torch.randn(100, 1),
        "columns": {"input": ["x1", "x2", "x3", "x4"], "target": ["y"]},
    }


@pytest.fixture
def sample_csv_file(tmp_path):
    """Write a temp CSV and point settings.upload_dir at it. Returns file_id."""
    torch.manual_seed(42)
    n, n_features = 200, 4
    X = torch.randn(n, n_features)
    y = (X[:, 0] * 2 + X[:, 1] * 0.5 + torch.randn(n) * 0.1).unsqueeze(1)

    original_upload_dir = settings.upload_dir
    settings.upload_dir = tmp_path
    file_id = "test_data.csv"
    csv_path = tmp_path / file_id

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "x3", "x4", "target"])
        for i in range(n):
            writer.writerow([
                f"{X[i, 0].item():.6f}",
                f"{X[i, 1].item():.6f}",
                f"{X[i, 2].item():.6f}",
                f"{X[i, 3].item():.6f}",
                f"{y[i, 0].item():.6f}",
            ])

    yield file_id

    settings.upload_dir = original_upload_dir


@pytest.fixture
def pipeline_config(sample_csv_file):
    """Minimal pipeline config for integration tests."""
    return {
        "file_id": sample_csv_file,
        "input_columns": "x1,x2,x3,x4",
        "target_columns": "target",
        "epochs": 3,
        "batch_size": 32,
        "val_ratio": 0.2,
        "shuffle": True,
        "loss_fn": "MSELoss",
        "optimizer": "Adam",
        "lr": 0.001,
    }
