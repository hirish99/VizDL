"""Stress tests for large dataset handling.

Run with: pytest tests/test_large_data.py -v -m slow
"""
import csv
from pathlib import Path

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from app.config import settings
from app.nodes.data import CSVLoaderNode, DataSplitterNode, _CHUNK_THRESHOLD_BYTES


pytestmark = pytest.mark.slow


def _write_csv(path: Path, n_rows: int, n_features: int):
    """Write a synthetic CSV with n_rows and n_features input columns + 1 target."""
    torch.manual_seed(42)
    cols = [f"x{i}" for i in range(n_features)] + ["target"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for _ in range(n_rows):
            row = [f"{torch.randn(1).item():.6f}" for _ in range(n_features + 1)]
            writer.writerow(row)
    return cols[:-1], ["target"]


class TestMediumDataset:
    """10K rows, 50 features — should complete quickly."""

    def test_csv_load_and_split(self, tmp_path):
        csv_path = tmp_path / "medium.csv"
        input_cols, target_cols = _write_csv(csv_path, 10_000, 50)

        original_dir = settings.upload_dir
        settings.upload_dir = tmp_path
        try:
            node = CSVLoaderNode()
            dataset = node.execute(
                file_id="medium.csv",
                input_columns=",".join(input_cols),
                target_columns=",".join(target_cols),
            )[0]

            assert dataset["X"].shape == (10_000, 50)
            assert dataset["y"].shape == (10_000, 1)

            splitter = DataSplitterNode()
            train_loader, val_loader = splitter.execute(
                dataset=dataset, val_ratio=0.2, batch_size=256,
            )

            # Verify data iteration works
            batch_count = 0
            for X_batch, y_batch in train_loader:
                assert X_batch.shape[1] == 50
                batch_count += 1
            assert batch_count > 0
        finally:
            settings.upload_dir = original_dir


class TestLargeDataset:
    """100K rows, 100 features — tests memory efficiency."""

    def test_csv_load_and_split(self, tmp_path):
        csv_path = tmp_path / "large.csv"
        input_cols, target_cols = _write_csv(csv_path, 100_000, 100)

        original_dir = settings.upload_dir
        settings.upload_dir = tmp_path
        try:
            node = CSVLoaderNode()
            dataset = node.execute(
                file_id="large.csv",
                input_columns=",".join(input_cols),
                target_columns=",".join(target_cols),
            )[0]

            assert dataset["X"].shape == (100_000, 100)
            assert dataset["y"].shape == (100_000, 1)

            splitter = DataSplitterNode()
            train_loader, val_loader = splitter.execute(
                dataset=dataset, val_ratio=0.2, batch_size=512,
            )

            # Count total samples via iteration
            total_train = sum(len(batch[0]) for batch in train_loader)
            total_val = sum(len(batch[0]) for batch in val_loader)
            assert total_train + total_val == 100_000
        finally:
            settings.upload_dir = original_dir


class TestSubsetMemoryEfficiency:
    """Verify Subset-based splitting doesn't copy data."""

    def test_subset_shares_underlying_tensors(self):
        X = torch.randn(1000, 10)
        y = torch.randn(1000, 1)
        full_ds = TensorDataset(X, y)

        indices = list(range(800))
        subset = Subset(full_ds, indices)

        # Subset wraps the original dataset, no copy
        assert subset.dataset is full_ds
        # Accessing an element should return data from the same tensors
        x_sub, y_sub = subset[0]
        assert torch.equal(x_sub, X[indices[0]])


class TestChunkedCSVReading:
    """Verify chunked reading produces identical results to non-chunked."""

    def test_chunked_vs_direct(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        input_cols, target_cols = _write_csv(csv_path, 5_000, 10)

        original_dir = settings.upload_dir
        settings.upload_dir = tmp_path
        try:
            # Load with standard approach
            node = CSVLoaderNode()
            dataset = node.execute(
                file_id="test.csv",
                input_columns=",".join(input_cols),
                target_columns=",".join(target_cols),
            )[0]
            X_direct = dataset["X"]
            y_direct = dataset["y"]

            # Now force chunked loading by temporarily lowering threshold
            import app.nodes.data as data_module
            original_threshold = data_module._CHUNK_THRESHOLD_BYTES
            data_module._CHUNK_THRESHOLD_BYTES = 0  # force chunked
            try:
                dataset2 = node.execute(
                    file_id="test.csv",
                    input_columns=",".join(input_cols),
                    target_columns=",".join(target_cols),
                )[0]
                X_chunked = dataset2["X"]
                y_chunked = dataset2["y"]
            finally:
                data_module._CHUNK_THRESHOLD_BYTES = original_threshold

            assert torch.equal(X_direct, X_chunked)
            assert torch.equal(y_direct, y_chunked)
        finally:
            settings.upload_dir = original_dir
