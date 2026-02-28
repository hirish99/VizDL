"""Hash-based execution cache for node outputs."""
import hashlib
import json
from typing import Any


class ExecutionCache:
    """Caches node outputs keyed by (node_type, params, input_hashes)."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _make_key(self, node_type: str, params: dict, input_keys: dict[str, str]) -> str:
        raw = json.dumps(
            {"type": node_type, "params": params, "inputs": input_keys},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, node_type: str, params: dict, input_keys: dict[str, str]) -> Any | None:
        key = self._make_key(node_type, params, input_keys)
        return self._cache.get(key)

    def put(self, node_type: str, params: dict, input_keys: dict[str, str], result: Any) -> str:
        key = self._make_key(node_type, params, input_keys)
        self._cache[key] = result
        return key

    def clear(self):
        self._cache.clear()
