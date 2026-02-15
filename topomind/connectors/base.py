from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


# ============================================================
# Abstract Execution Connector
# ============================================================

class ExecutionConnector(ABC):
    """
    Abstract execution backend representing the boundary between
    agent cognition and external deterministic systems.
    """

    @property
    def name(self) -> str:
        """Return connector identity (useful for telemetry/logging)."""
        return self.__class__.__name__

    @abstractmethod
    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        timeout: int,
    ) -> Any:
        """
        Execute a tool call within the specified timeout.

        Implementations must:
            • Be deterministic for identical inputs
            • Respect timeout
            • Raise TimeoutError on timeout
            • Raise Exception on deterministic failure
            • Never mutate input args
        """
        if timeout <= 0:
            raise ValueError("Timeout must be positive integer")

        raise NotImplementedError

    # ---------------------------------------------------------
    # Optional Lifecycle Hooks
    # ---------------------------------------------------------

    def health(self) -> bool:
        """
        Return connector health status.
        Default implementation assumes healthy.
        """
        return True

    def shutdown(self) -> None:
        """
        Gracefully release resources (connections, sessions, etc.).
        """
        pass


# ============================================================
# Fake Connector (Testing)
# ============================================================

class FakeConnector(ExecutionConnector):
    """
    Deterministic testing connector.

    Respects tool output expectations.
    """

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        timeout: int,
    ) -> Any:

        if timeout <= 0:
            raise ValueError("Timeout must be positive integer")

        # Defensive copy (never mutate caller input)
        safe_args = dict(args)

        if tool_name == "echo":
            return {"text": safe_args.get("text", "")}

        # Simulate unknown tool failure
        raise Exception(f"Unknown tool: {tool_name}")
