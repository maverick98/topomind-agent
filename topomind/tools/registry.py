from __future__ import annotations

from typing import Dict, List, Any, Iterable
from threading import RLock
from copy import deepcopy

from .schema import Tool


class ToolRegistry:
    """
    Authoritative registry of all tools available to the agent.

    This forms the capability boundary: if a tool is not registered here,
    it is not executable by the agent.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._lock = RLock()  # Future-proof for concurrent planners/executors

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Raises
        ------
        ValueError
            If tool name already exists or is invalid.
        """
        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("Tool must have a valid string name.")

        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' is already registered.")
            self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        """Register multiple tools atomically."""
        with self._lock:
            for tool in tools:
                if not tool.name or not isinstance(tool.name, str):
                    raise ValueError("Tool must have a valid string name.")
                if tool.name in self._tools:
                    raise ValueError(f"Tool '{tool.name}' is already registered.")

            for tool in tools:
                self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, tool_name: str) -> Tool:
        """Retrieve a tool by name."""
        with self._lock:
            try:
                return self._tools[tool_name]
            except KeyError:
                raise KeyError(f"Tool '{tool_name}' is not registered.") from None

    def has_tool(self, tool_name: str) -> bool:
        """Check whether a tool exists."""
        with self._lock:
            return tool_name in self._tools

    def list_tools(self) -> List[Tool]:
        """
        Return a copy of registered tools to prevent external mutation.
        Deterministically sorted for planner stability.
        """
        with self._lock:
            return sorted(self._tools.values(), key=lambda t: t.name)

    def list_tool_names(self) -> List[str]:
        """Return all registered tool names."""
        with self._lock:
            return sorted(self._tools.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    # ------------------------------------------------------------------
    # Schema Access (Contract Authority)
    # ------------------------------------------------------------------

    def get_input_schema(self, tool_name: str) -> Dict[str, Any]:
        """Return declared input schema (copy-safe)."""
        return deepcopy(self.get(tool_name).input_schema)

    def get_output_schema(self, tool_name: str) -> Dict[str, Any]:
        """Return declared output schema (copy-safe)."""
        return deepcopy(self.get(tool_name).output_schema)

    # ------------------------------------------------------------------
    # Planner Integration (NEW – Non-Breaking)
    # ------------------------------------------------------------------

    def get_planner_manifest(self) -> List[Dict[str, Any]]:
        """
        Returns tool definitions formatted for planner consumption.

        Includes:
        - name
        - description
        - input_schema
        - prompt (if defined)
        - strict flag
        """
        with self._lock:
            manifest: List[Dict[str, Any]] = []

            for tool in sorted(self._tools.values(), key=lambda t: t.name):
                manifest.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": deepcopy(tool.input_schema),
                    "prompt": tool.prompt,
                    "strict": tool.strict,
                    "version": tool.version,
                    "timeout_seconds": tool.timeout_seconds,
                    "retryable": tool.retryable,
                    "side_effect": tool.side_effect,
                    "tags": list(tool.tags),
                })

            return manifest

    def get_strict_tools(self) -> List[Tool]:
        """
        Returns tools marked as strict.
        Useful for planner-level enforcement.
        """
        with self._lock:
            return [t for t in self._tools.values() if t.strict]

    def has_strict_tools(self) -> bool:
        """Returns True if any registered tool is strict."""
        with self._lock:
            return any(t.strict for t in self._tools.values())
