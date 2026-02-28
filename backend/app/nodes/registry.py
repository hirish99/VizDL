"""Node registry with auto-discovery."""
import importlib
import pkgutil
from typing import Any

from .base import BaseNode, NodeDefinition


class NodeRegistry:
    """Singleton registry mapping node type strings to BaseNode subclasses."""

    _nodes: dict[str, type[BaseNode]] = {}

    @classmethod
    def register(cls, node_type: str | None = None):
        """Decorator to register a node class.

        Usage:
            @NodeRegistry.register()
            class MyNode(BaseNode):
                ...

            @NodeRegistry.register("custom_name")
            class MyNode(BaseNode):
                ...
        """
        def decorator(node_cls: type[BaseNode]) -> type[BaseNode]:
            name = node_type or node_cls.__name__
            cls._nodes[name] = node_cls
            return node_cls
        return decorator

    @classmethod
    def get(cls, node_type: str) -> type[BaseNode]:
        if node_type not in cls._nodes:
            raise KeyError(f"Unknown node type: {node_type}")
        return cls._nodes[node_type]

    @classmethod
    def create(cls, node_type: str) -> BaseNode:
        return cls.get(node_type)()

    @classmethod
    def all_definitions(cls) -> dict[str, NodeDefinition]:
        return {
            name: node_cls.get_definition(name)
            for name, node_cls in cls._nodes.items()
        }

    @classmethod
    def discover(cls, package_name: str) -> None:
        """Import all modules in the given package to trigger @register decorators."""
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return
        if not hasattr(package, "__path__"):
            return
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            if module_name.startswith("_") or module_name in ("base", "registry"):
                continue
            importlib.import_module(f"{package_name}.{module_name}")

    @classmethod
    def clear(cls) -> None:
        cls._nodes.clear()
