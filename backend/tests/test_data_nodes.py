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
        cols = ["feat_0", "feat_1", "feat_2", "coord_x", "y"]
        assert _resolve_columns(["feat_*"], cols) == ["feat_0", "feat_1", "feat_2"]

    def test_glob_preserves_order(self):
        cols = ["b_0", "a_0", "b_1", "a_1", "target"]
        assert _resolve_columns(["a_*"], cols) == ["a_0", "a_1"]

    def test_mixed_exact_and_glob(self):
        cols = ["feat_0", "feat_1", "coord_x", "coord_y", "y"]
        result = _resolve_columns(["feat_*", "coord_x", "coord_y"], cols)
        assert result == ["feat_0", "feat_1", "coord_x", "coord_y"]

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
        csv_path = tmp_path / "features.csv"
        header = [f"feat_{i}" for i in range(10)] + ["coord_x", "coord_y", "target"]
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
                file_id="features.csv",
                input_columns="feat_*,coord_x,coord_y",
                target_columns="target",
            )[0]

            assert dataset["X"].shape == (5, 12)  # 10 features + 2 coords
            assert dataset["y"].shape == (5, 1)
            assert len(dataset["columns"]["input"]) == 12
            assert dataset["columns"]["target"] == ["target"]
        finally:
            settings.upload_dir = original_dir
