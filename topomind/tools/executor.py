from __future__ import annotations

import time
from typing import Dict, Any

from .registry import ToolRegistry
from ..connectors.manager import ConnectorManager
from ..models import ToolResult
from .validator import ArgumentValidator, ArgumentValidationError
from .output_validator import OutputValidator, OutputValidationError


class ToolExecutor:
    """
    Core execution kernel responsible for running agent tools safely and deterministically.

    This component forms the controlled boundary between agent cognition
    (planning and reasoning) and real-world execution via connectors.

    Execution Flow
    --------------
    Planner → ToolCall → ArgumentValidator → Connector → OutputValidator → ToolResult

    Responsibilities
    ----------------
    • Resolve tool and connector from registries  
    • Enforce argument schema validation (input firewall)  
    • Enforce output schema validation (output firewall)  
    • Apply execution policies (timeout, retries)  
    • Isolate failures from propagating into agent logic  
    • Measure execution latency using a monotonic clock  
    • Produce structured, version-aware ToolResult objects  

    Safety Guarantees
    -----------------
    • Hallucinated parameters are rejected before execution  
    • Malformed tool outputs are blocked from entering memory  
    • Connector failures do not crash the agent loop  
    • Tool contract version is preserved for schema evolution  

    This class contains no business logic and no external system
    interactions beyond delegating to connectors.
    """

    def __init__(self, registry: ToolRegistry, connectors: ConnectorManager) -> None:
        """
        Parameters
        ----------
        registry : ToolRegistry
            Source of tool contracts and schemas.

        connectors : ConnectorManager
            Execution backend routing layer.
        """
        self._registry = registry
        self._connectors = connectors
        self._arg_validator = ArgumentValidator(registry)
        self._out_validator = OutputValidator(registry)

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool call under controlled runtime policies.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute.

        args : Dict[str, Any]
            Proposed input arguments (validated before execution).

        Returns
        -------
        ToolResult
            Immutable execution result with version, latency,
            status classification, and stability signal.
        """
        start = time.monotonic()

        try:
            tool = self._registry.get(tool_name)
            connector = self._connectors.get(tool.connector_name)
        except KeyError as e:
            return self._blocked_result(tool_name, str(e))
        except Exception as e:
            return self._blocked_result(tool_name, f"Registry failure: {e}")

        # Argument Validation
        try:
            args = self._arg_validator.validate(tool_name, args)
        except ArgumentValidationError as e:
            return self._failure_result(tool_name, tool.version, f"Invalid arguments: {e}", start)

        max_attempts = tool.max_retries + 1 if tool.retryable else 1
        timeout = tool.timeout_seconds

        for attempt in range(max_attempts):
            try:
                raw_output = self._execute_with_timeout(connector, tool_name, args, timeout)

                # Output Validation
                output = self._out_validator.validate(tool_name, raw_output)

                stability = 1.0 - (attempt * 0.1)
                return self._success_result(tool_name, tool.version, output, start, stability)

            except OutputValidationError as e:
                return self._failure_result(tool_name, tool.version, f"Invalid output: {e}", start)

            except TimeoutError:
                error = f"Execution timed out after {timeout}s"

            except Exception as e:
                error = str(e)

            if attempt >= max_attempts - 1:
                return self._failure_result(tool_name, tool.version, error, start)

        return self._failure_result(tool_name, tool.version, "Unknown execution state", start)

    def _execute_with_timeout(self, connector, tool_name: str, args: Dict[str, Any], timeout: int) -> Any:
        """
        Delegate execution to connector while enforcing timeout policy.
        """
        return connector.execute(tool_name, args, timeout=timeout)

    # ------------------------------------------------------------------
    # Result Builders
    # ------------------------------------------------------------------

    def _success_result(self, tool_name: str, tool_version: str, output: Any, start_time: float, stability: float) -> ToolResult:
        """Build a successful execution result."""
        return ToolResult(
            tool_name=tool_name,
            tool_version=tool_version,
            status="success",
            output=output,
            error=None,
            latency_ms=self._latency_ms(start_time),
            stability_signal=max(0.0, min(1.0, stability)),
        )

    def _failure_result(self, tool_name: str, tool_version: str, error: str, start_time: float) -> ToolResult:
        """Build a failure result caused during execution."""
        return ToolResult(
            tool_name=tool_name,
            tool_version=tool_version,
            status="failure",
            output=None,
            error=error,
            latency_ms=self._latency_ms(start_time),
            stability_signal=0.0,
        )

    def _blocked_result(self, tool_name: str, error: str) -> ToolResult:
        """Build a blocked result (policy or resolution failure)."""
        return ToolResult(
            tool_name=tool_name,
            tool_version="unknown",
            status="blocked",
            output=None,
            error=error,
            latency_ms=0,
            stability_signal=0.0,
        )
    # ------------------------------------------------------------------
    # Public Accessors
    # ------------------------------------------------------------------

    @property
    def registry(self) -> ToolRegistry:
        """
        Read-only access to tool registry.

        The Agent uses this to provide available tools to the planner.
        Execution authority remains inside ToolExecutor.
        """
        return self._registry
    @staticmethod
    def _latency_ms(start_time: float) -> int:
        """Return elapsed time in milliseconds using a monotonic clock."""
        return int((time.monotonic() - start_time) * 1000)
