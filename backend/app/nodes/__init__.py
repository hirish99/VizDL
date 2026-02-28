"""Auto-discover all node modules on import."""
from .registry import NodeRegistry

NodeRegistry.discover("app.nodes")
