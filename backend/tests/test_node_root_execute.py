"""Test that every architecture node can execute as a root node (no upstream inputs).

The VRAM estimator calls node.execute() with only param kwargs and no handle inputs
when a node has no incoming edges. Every arch node must handle this gracefully by
using kwargs.get() instead of kwargs[] for handle inputs.
"""
import pytest

from app.nodes.registry import NodeRegistry
from app.nodes.base import DataType


# Ensure all nodes are discovered
NodeRegistry.discover("app.nodes")


def _is_arch_node(node_cls):
    """Check if a node produces ARCH outputs (layer/structural nodes)."""
    try:
        outputs = node_cls.RETURN_TYPES()
        return any(o.dtype == DataType.ARCH for o in outputs)
    except Exception:
        return False


def _get_arch_node_types():
    """Return all registered node types that produce ARCH outputs."""
    return [
        nt for nt, cls in NodeRegistry._nodes.items()
        if _is_arch_node(cls)
    ]


def _get_default_params(node_cls):
    """Build a kwargs dict with only the non-handle params using their defaults."""
    input_types = node_cls.INPUT_TYPES()
    kwargs = {}
    for name, spec in input_types.items():
        if not spec.is_handle and spec.default is not None:
            kwargs[name] = spec.default
        elif not spec.is_handle and spec.required:
            if spec.dtype == DataType.INT:
                kwargs[name] = spec.min_val or 1
            elif spec.dtype == DataType.FLOAT:
                kwargs[name] = 0.0
            elif spec.dtype == DataType.STRING:
                if spec.choices:
                    kwargs[name] = spec.choices[0]
                else:
                    kwargs[name] = ""
            elif spec.dtype == DataType.BOOL:
                kwargs[name] = False
    return kwargs


@pytest.mark.parametrize("node_type", _get_arch_node_types())
def test_node_executes_without_upstream(node_type):
    """Every architecture node must not crash when called with no handle inputs."""
    node_cls = NodeRegistry.get(node_type)
    node = node_cls()
    node._node_id = f"test_{node_type}"
    kwargs = _get_default_params(node_cls)
    result = node.execute(**kwargs)
    assert isinstance(result, tuple)
    assert len(result) >= 1


@pytest.mark.parametrize("node_type", _get_arch_node_types())
def test_node_on_disable_without_upstream(node_type):
    """Every architecture node's on_disable must not crash with no handle inputs."""
    node_cls = NodeRegistry.get(node_type)
    node = node_cls()
    node._node_id = f"test_{node_type}"
    kwargs = _get_default_params(node_cls)
    result = node.on_disable(**kwargs)
    assert isinstance(result, tuple)
