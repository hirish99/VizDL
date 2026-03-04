"""Tests for data nodes: CSVLoader glob column patterns and basic loading."""
import csv
from pathlib import Path

import pytest
import torch

from app.config import settings
from app.nodes.data import CSVLoaderNode, _resolve_columns


class TestResolveColumns:
    """Unit tests for _resolve_columns glob expansion."""

    def test_exact_match(self):
        assert _resolve_columns(["x1", "x2"], ["x1", "x2", "y"]) == ["x1", "x2"]

    def test_glob_star(self):
        cols = ["s_0_g", "s_1_g", "s_2_g", "query_x", "u"]
        assert _resolve_columns(["s_*_g"], cols) == ["s_0_g", "s_1_g", "s_2_g"]

    def test_glob_preserves_order(self):
        cols = ["b_0", "a_0", "b_1", "a_1", "target"]
        assert _resolve_columns(["a_*"], cols) == ["a_0", "a_1"]

    def test_mixed_exact_and_glob(self):
        cols = ["s_0_g", "s_1_g", "query_x", "query_y", "u"]
        result = _resolve_columns(["s_*_g", "query_x", "query_y"], cols)
        assert result == ["s_0_g", "s_1_g", "query_x", "query_y"]

    def test_no_duplicates(self):
        cols = ["x1", "x2", "x3"]
        result = _resolve_columns(["x1", "x*"], cols)
        assert result == ["x1", "x2", "x3"]

    def test_question_mark_glob(self):
        cols = ["x1", "x2", "x10", "y"]
        assert _resolve_columns(["x?"], cols) == ["x1", "x2"]

    def test_missing_exact_raises(self):
        with pytest.raises(ValueError, match="not found"):
            _resolve_columns(["missing"], ["x1", "x2"])

    def test_empty_glob_raises(self):
        with pytest.raises(ValueError, match="matched no columns"):
            _resolve_columns(["z_*"], ["x1", "x2"])


class TestCSVLoaderGlob:
    """Integration test: CSVLoader with glob patterns on a real CSV."""

    def test_glob_columns_load(self, tmp_path):
        csv_path = tmp_path / "sensors.csv"
        header = [f"s_{i}_g" for i in range(10)] + ["query_x", "query_y", "u"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for _ in range(5):
                writer.writerow([0.1] * len(header))

        original_dir = settings.upload_dir
        settings.upload_dir = tmp_path
        try:
            node = CSVLoaderNode()
            dataset = node.execute(
                file_id="sensors.csv",
                input_columns="s_*_g,query_x,query_y",
                target_columns="u",
            )[0]

            assert dataset["X"].shape == (5, 12)  # 10 sensors + 2 query coords
            assert dataset["y"].shape == (5, 1)
            assert len(dataset["columns"]["input"]) == 12
            assert dataset["columns"]["target"] == ["u"]
        finally:
            settings.upload_dir = original_dir


class TestPTLoader:
    """Integration test: CSVLoader loading .pt files."""

    def test_pt_load_basic(self, tmp_path):
        header = ["x1", "x2", "target"]
        data = torch.randn(100, 3)
        torch.save({"data": data, "header": header}, tmp_path / "test.pt")

        original_dir = settings.upload_dir
        settings.upload_dir = tmp_path
        try:
            node = CSVLoaderNode()
            dataset = node.execute(
                file_id="test.pt",
                input_columns="x1,x2",
                target_columns="target",
            )[0]

            assert dataset["X"].shape == (100, 2)
            assert dataset["y"].shape == (100, 1)
            assert torch.allclose(dataset["X"], data[:, :2])
            assert torch.allclose(dataset["y"], data[:, 2:3])
        finally:
            settings.upload_dir = original_dir

    def test_pt_glob_columns(self, tmp_path):
        header = [f"s_{i}_g" for i in range(10)] + ["query_x", "query_y", "u"]
        data = torch.randn(50, 13)
        torch.save({"data": data, "header": header}, tmp_path / "sensors.pt")

        original_dir = settings.upload_dir
        settings.upload_dir = tmp_path
        try:
            node = CSVLoaderNode()
            dataset = node.execute(
                file_id="sensors.pt",
                input_columns="s_*_g,query_x,query_y",
                target_columns="u",
            )[0]

            assert dataset["X"].shape == (50, 12)
            assert dataset["y"].shape == (50, 1)
            assert len(dataset["columns"]["input"]) == 12
            assert torch.allclose(dataset["X"][:, :10], data[:, :10])
        finally:
            settings.upload_dir = original_dir
