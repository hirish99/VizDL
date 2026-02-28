"""Data nodes: CSV loading and train/val splitting."""
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry
from ..config import settings

# Files larger than this use chunked reading to reduce peak memory
_CHUNK_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB


def _resolve_columns(specs: list[str], available: list[str]) -> list[str]:
    """Expand column specs against available headers.

    Each spec can be:
      - An exact column name: ``x1``
      - A glob pattern: ``feature_*`` (matches via fnmatch)

    Returns resolved column names in the order specs appear,
    with glob matches sorted by their position in `available`.
    Raises ValueError if a non-glob spec doesn't match any column.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if "*" in spec or "?" in spec or "[" in spec:
            matches = [c for c in available if fnmatch(c, spec) and c not in seen]
            if not matches:
                raise ValueError(f"Column pattern '{spec}' matched no columns")
            resolved.extend(matches)
            seen.update(matches)
        else:
            if spec not in available:
                raise ValueError(
                    f"Column '{spec}' not found. Available: {available[:10]}{'...' if len(available) > 10 else ''}"
                )
            if spec not in seen:
                resolved.append(spec)
                seen.add(spec)
    return resolved


@NodeRegistry.register("CSVLoader")
class CSVLoaderNode(BaseNode):
    CATEGORY = "Data"
    DISPLAY_NAME = "CSV Loader"
    DESCRIPTION = "Load a CSV file and select input/target columns (supports glob patterns like feature_*)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "file_id": InputSpec(dtype=DataType.STRING, required=True, is_handle=False),
            "input_columns": InputSpec(
                dtype=DataType.STRING, required=True, is_handle=False,
                default="x1,x2",
            ),
            "target_columns": InputSpec(
                dtype=DataType.STRING, required=True, is_handle=False,
                default="target",
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.DATASET, name="dataset")]

    def execute(self, **kwargs) -> tuple:
        file_id = kwargs["file_id"]
        file_path = settings.upload_dir / file_id
        if not file_path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {file_id}")

        raw_input = [c.strip() for c in kwargs["input_columns"].split(",") if c.strip()]
        raw_target = [c.strip() for c in kwargs["target_columns"].split(",") if c.strip()]

        if not raw_input:
            raise ValueError("No input columns specified")
        if not raw_target:
            raise ValueError("No target columns specified")

        # Read header to resolve glob patterns
        header = list(pd.read_csv(file_path, nrows=0).columns)
        input_cols = _resolve_columns(raw_input, header)
        target_cols = _resolve_columns(raw_target, header)

        file_size = file_path.stat().st_size
        if file_size < _CHUNK_THRESHOLD_BYTES:
            df = pd.read_csv(file_path)
            X = torch.tensor(df[input_cols].values, dtype=torch.float32)
            y = torch.tensor(df[target_cols].values, dtype=torch.float32)
        else:
            chunks_X, chunks_y = [], []
            for chunk in pd.read_csv(file_path, chunksize=50_000):
                chunks_X.append(torch.tensor(chunk[input_cols].values, dtype=torch.float32))
                chunks_y.append(torch.tensor(chunk[target_cols].values, dtype=torch.float32))
            X = torch.cat(chunks_X)
            y = torch.cat(chunks_y)
            del chunks_X, chunks_y

        return ({"X": X, "y": y, "columns": {"input": input_cols, "target": target_cols}},)


@NodeRegistry.register("DataSplitter")
class DataSplitterNode(BaseNode):
    CATEGORY = "Data"
    DISPLAY_NAME = "Data Splitter"
    DESCRIPTION = "Split dataset into train and validation sets"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "dataset": InputSpec(dtype=DataType.DATASET, required=True),
            "val_ratio": InputSpec(
                dtype=DataType.FLOAT, default=0.2, required=False,
                min_val=0.01, max_val=0.99, is_handle=False,
            ),
            "batch_size": InputSpec(
                dtype=DataType.INT, default=32, required=False,
                min_val=1, is_handle=False,
            ),
            "shuffle": InputSpec(
                dtype=DataType.BOOL, default=True, required=False,
                is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [
            OutputSpec(dtype=DataType.DATASET, name="train_loader"),
            OutputSpec(dtype=DataType.DATASET, name="val_loader"),
        ]

    def execute(self, **kwargs) -> tuple:
        dataset = kwargs["dataset"]
        val_ratio = kwargs.get("val_ratio", 0.2)
        batch_size = kwargs.get("batch_size", 32)
        shuffle = kwargs.get("shuffle", True)

        X, y = dataset["X"], dataset["y"]
        n = len(X)
        n_val = max(1, int(n * val_ratio))

        indices = torch.randperm(n).tolist()
        full_ds = TensorDataset(X, y)
        train_ds = Subset(full_ds, indices[n_val:])
        val_ds = Subset(full_ds, indices[:n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return (train_loader, val_loader)
