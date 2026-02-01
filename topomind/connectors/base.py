from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class ExecutionConnector(ABC):
    """
    Abstract execution backend representing the boundary between
    agent cognition and external deterministic systems.

    A connector is responsible for performing real-world actions such as:
        • Calling APIs
        • Querying databases
        • Running local compute
        • Interacting with enterprise systems

    Architectural Role
    -------------------
    ToolExecutor enforces *policy* (validation, retries, timeouts).
    ExecutionConnector performs the *actual operation*.

    Connectors must:
        • Be deterministic given the same inputs
        • Respect the provided timeout
        • Raise TimeoutError on timeout
        • Raise standard Exceptions for execution failures
        • Never mutate the provided arguments
        • Avoid leaking resources across calls
    """

    @abstractmethod
    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        timeout: int,
    ) -> Any:
        """
        Execute a tool call within the specified timeout.

        Parameters
        ----------
        tool_name : str
            Name of the tool being invoked.

        args : Dict[str, Any]
            Validated input arguments. MUST NOT be mutated.

        timeout : int
            Maximum allowed execution time in seconds.

        Returns
        -------
        Any
            Structured tool output matching the tool's declared output schema.

        Raises
        ------
        TimeoutError
            If execution exceeds the allowed time.

        Exception
            For deterministic execution failures (network errors,
            database failures, etc.).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional Lifecycle Hooks
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """
        Return the connector's health status.

        Used by stability systems to detect degraded or failing backends.
        Default implementation assumes healthy.
        """
        return True

    def shutdown(self) -> None:
        """
        Gracefully release resources (connections, sessions, etc.).

        Called during agent shutdown or connector replacement.
        """
        pass


class FakeConnector(ExecutionConnector):
    """Testing connector that respects tool output schema."""

    def execute(self, tool_name: str, args: Dict[str, Any], timeout: int) -> Any:
        if tool_name == "echo":
            # Must return dict to match output_schema
            return {"text": args.get("text", "")}

        return {"result": f"Tool {tool_name} executed with args {args}"}
