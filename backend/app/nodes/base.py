"""Base node abstraction and data type definitions."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataType(str, Enum):
    TENSOR = "TENSOR"
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOL = "BOOL"
    DATASET = "DATASET"
    MODEL = "MODEL"
    OPTIMIZER = "OPTIMIZER"
    LOSS_FN = "LOSS_FN"
    METRICS = "METRICS"
    TRAINING_RESULT = "TRAINING_RESULT"
    LAYER_SPEC = "LAYER_SPEC"
    LAYER_SPECS = "LAYER_SPECS"
    ARCH = "ARCH"
    ANY = "ANY"


# Which types can connect to which
TYPE_COMPATIBILITY: dict[DataType, set[DataType]] = {
    dt: {dt, DataType.ANY} for dt in DataType
}
TYPE_COMPATIBILITY[DataType.ANY] = set(DataType)
TYPE_COMPATIBILITY[DataType.LAYER_SPEC].add(DataType.LAYER_SPECS)


@dataclass
class InputSpec:
    dtype: DataType
    default: Any = None
    required: bool = True
    min_val: float | None = None
    max_val: float | None = None
    choices: list[Any] | None = None
    is_handle: bool = True  # True = comes from edge; False = set in properties


@dataclass
class OutputSpec:
    dtype: DataType
    name: str


@dataclass
class NodeDefinition:
    """Serializable node definition sent to the frontend."""
    node_type: str
    display_name: str
    category: str
    description: str
    inputs: dict[str, InputSpec]
    outputs: list[OutputSpec]


class BaseNode(ABC):
    """Abstract base class for all nodes in the graph."""

    CATEGORY: str = "Uncategorized"
    DISPLAY_NAME: str = ""
    DESCRIPTION: str = ""

    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls) -> dict[str, InputSpec]:
        ...

    @classmethod
    @abstractmethod
    def RETURN_TYPES(cls) -> list[OutputSpec]:
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> tuple[Any, ...]:
        ...

    def on_disable(self, **kwargs: Any) -> tuple[Any, ...]:
        """Default ablation bypass: identity passthrough on first input."""
        for v in kwargs.values():
            return (v,)
        return (None,)

    @classmethod
    def get_definition(cls, node_type: str) -> NodeDefinition:
        return NodeDefinition(
            node_type=node_type,
            display_name=cls.DISPLAY_NAME or cls.__name__,
            category=cls.CATEGORY,
            description=cls.DESCRIPTION or cls.__doc__ or "",
            inputs=cls.INPUT_TYPES(),
            outputs=cls.RETURN_TYPES(),
        )
