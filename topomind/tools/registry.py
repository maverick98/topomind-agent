from __future__ import annotations

from typing import Dict, List, Any, Iterable
from threading import RLock
from copy import deepcopy
import logging

from .schema import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Authoritative registry of all tools available to the agent.

    This forms the capability boundary: if a tool is not registered here,
    it is not executable by the agent.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._lock = RLock()
        logger.info("[TOOL REGISTRY] Initialized (empty)")

    # ------------------------------------------------------------------
    # Strict Registration (Original Behavior Preserved)
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:

        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("Tool must have a valid string name.")

        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' is already registered.")

            self._tools[tool.name] = tool

            logger.info(
                "[TOOL REGISTRY] Tool registered | total=%d",
                len(self._tools)
            )

            logger.info(tool.to_debug_string())

    # ------------------------------------------------------------------
    # Idempotent Registration (NEW)
    # ------------------------------------------------------------------

    def register_or_update(self, tool: Tool) -> str:
        """
        Idempotent registration.
        - If tool does not exist → register
        - If tool exists and contract unchanged → no-op
        - If tool exists and contract changed → replace

        Returns:
            "registered" | "updated" | "unchanged"
        """

        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("Tool must have a valid string name.")

        with self._lock:
            existing = self._tools.get(tool.name)

            # First-time registration
            if existing is None:
                self._tools[tool.name] = tool

                logger.info(
                    "[TOOL REGISTRY] Tool registered | total=%d",
                    len(self._tools)
                )
                logger.info(tool.to_debug_string())

                return "registered"

            # Contract unchanged → no-op
            if existing.contract_hash == tool.contract_hash:
                logger.info(
                    "[TOOL REGISTRY] Tool unchanged: %s",
                    tool.name
                )
                return "unchanged"

            # Contract changed → update
            self._tools[tool.name] = tool

            logger.info(
                "[TOOL REGISTRY] Tool updated: %s",
                tool.name
            )
            logger.info(tool.to_debug_string())

            return "updated"

    # ------------------------------------------------------------------
    # Bulk Registration (Strict)
    # ------------------------------------------------------------------

    def register_many(self, tools: Iterable[Tool]) -> None:

        with self._lock:
            for tool in tools:
                if not tool.name or not isinstance(tool.name, str):
                    raise ValueError("Tool must have a valid string name.")
                if tool.name in self._tools:
                    raise ValueError(f"Tool '{tool.name}' is already registered.")

            for tool in tools:
                self._tools[tool.name] = tool
                logger.info("[TOOL REGISTRY] Bulk registered: %s", tool.name)

            logger.info(
                "[TOOL REGISTRY] Bulk registration complete | total=%d",
                len(self._tools)
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, tool_name: str) -> Tool:

        with self._lock:
            try:
                tool = self._tools[tool_name]
                logger.debug(
                    "[TOOL REGISTRY] Lookup success: %s | hash=%s",
                    tool_name,
                    tool.contract_hash
                )
                return tool
            except KeyError:
                logger.error(
                    "[TOOL REGISTRY] Lookup FAILED: %s | available=%s",
                    tool_name,
                    list(self._tools.keys())
                )
                raise KeyError(f"Tool '{tool_name}' is not registered.") from None

    def has_tool(self, tool_name: str) -> bool:
        with self._lock:
            exists = tool_name in self._tools
            logger.debug(
                "[TOOL REGISTRY] has_tool(%s) -> %s",
                tool_name,
                exists
            )
            return exists

    def list_tools(self) -> List[Tool]:

        with self._lock:
            tools = sorted(self._tools.values(), key=lambda t: t.name)

            logger.info(
                "[TOOL REGISTRY] list_tools | count=%d | names=%s",
                len(tools),
                [t.name for t in tools]
            )

            return tools

    def list_tool_names(self) -> List[str]:

        with self._lock:
            names = sorted(self._tools.keys())

            logger.info(
                "[TOOL REGISTRY] list_tool_names -> %s",
                names
            )

            return names

    def __len__(self) -> int:
        with self._lock:
            count = len(self._tools)
            logger.debug("[TOOL REGISTRY] __len__ -> %d", count)
            return count

    # ------------------------------------------------------------------
    # Schema Access
    # ------------------------------------------------------------------

    def get_input_schema(self, tool_name: str) -> Dict[str, Any]:
        return deepcopy(self.get(tool_name).input_schema)

    def get_output_schema(self, tool_name: str) -> Dict[str, Any]:
        return deepcopy(self.get(tool_name).output_schema)

    # ------------------------------------------------------------------
    # Planner Integration
    # ------------------------------------------------------------------

    def get_planner_manifest(self) -> List[Dict[str, Any]]:

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
                    "execution_model": tool.execution_model,
                    "timeout_seconds": tool.timeout_seconds,
                    "retryable": tool.retryable,
                    "side_effect": tool.side_effect,
                    "tags": list(tool.tags),
                    "produces": list(tool.produces),
                    "consumes": list(tool.consumes),
                    "contract_hash": tool.contract_hash,
                })

            logger.info(
                "[TOOL REGISTRY] Planner manifest generated | count=%d",
                len(manifest)
            )

            return manifest

    def get_strict_tools(self) -> List[Tool]:

        with self._lock:
            strict_tools = [t for t in self._tools.values() if t.strict]

            logger.info(
                "[TOOL REGISTRY] Strict tools | count=%d | names=%s",
                len(strict_tools),
                [t.name for t in strict_tools]
            )

            return strict_tools

    def has_strict_tools(self) -> bool:

        with self._lock:
            result = any(t.strict for t in self._tools.values())

            logger.info(
                "[TOOL REGISTRY] has_strict_tools -> %s",
                result
            )

            return result
