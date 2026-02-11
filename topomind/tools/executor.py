from __future__ import annotations

import time
import logging
from typing import Dict, Any

from .registry import ToolRegistry
from ..connectors.manager import ConnectorManager
from ..models import ToolResult
from .validator import ArgumentValidator, ArgumentValidationError
from .output_validator import OutputValidator, OutputValidationError

logger = logging.getLogger(__name__)


class ToolExecutor:

    def __init__(self, registry: ToolRegistry, connectors: ConnectorManager) -> None:
        self._registry = registry
        self._connectors = connectors
        self._arg_validator = ArgumentValidator(registry)
        self._out_validator = OutputValidator(registry)

    # ============================================================
    # MAIN EXECUTION
    # ============================================================

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:

        start = time.monotonic()

        # ------------------------------------------------------------
        # Resolve Tool + Connector
        # ------------------------------------------------------------
        try:
            tool = self._registry.get(tool_name)
            connector = self._connectors.get(tool.connector_name)
        except KeyError as e:
            return self._blocked_result(tool_name, str(e))
        except Exception as e:
            return self._blocked_result(tool_name, f"Registry failure: {e}")

        # ------------------------------------------------------------
        # Argument Validation
        # ------------------------------------------------------------
        try:
            args = self._arg_validator.validate(tool_name, args)
        except ArgumentValidationError as e:
            return self._failure_result(
                tool_name,
                tool.version,
                f"Invalid arguments: {e}",
                start,
            )

        max_attempts = tool.max_retries + 1 if tool.retryable else 1
        timeout = tool.timeout_seconds

        # ------------------------------------------------------------
        # Execution Loop
        # ------------------------------------------------------------
        for attempt in range(max_attempts):
            try:

                # ============================================================
                # CASE 1: Tool has execution_model (LLM involved)
                # ============================================================
                if tool.execution_model:

                    try:
                        llm_connector = self._connectors.get("llm")
                    except KeyError:
                        return self._failure_result(
                            tool_name,
                            tool.version,
                            "No 'llm' connector registered for execution_model",
                            start,
                        )

                    logger.info(
                        f"[EXECUTION MODEL] Using model: {tool.execution_model}"
                    )

                    generated_output = llm_connector.execute(
                        tool,
                        args,
                        timeout=timeout,
                    )

                    # --------------------------------------------------------
                    # If the tool itself uses LLM connector,
                    # then Phase 1 IS the final execution.
                    # --------------------------------------------------------
                    if tool.connector_name == "llm":
                        raw_output = generated_output

                    else:
                        # ----------------------------------------------------
                        # Hybrid mode: LLM → Deterministic connector
                        # ----------------------------------------------------
                        if isinstance(generated_output, dict):
                            args = generated_output

                        elif isinstance(generated_output, str):
                            if tool.output_schema and len(tool.output_schema) == 1:
                                key = next(iter(tool.output_schema))
                                args = {key: generated_output}
                            else:
                                args = generated_output
                        else:
                            raise RuntimeError(
                                "LLM execution_model must return string or dict"
                            )

                        raw_output = connector.execute(tool, args, timeout=timeout)

                # ============================================================
                # CASE 2: Pure deterministic tool
                # ============================================================
                else:
                    raw_output = connector.execute(tool, args, timeout=timeout)

                # ------------------------------------------------------------
                # Output Validation
                # ------------------------------------------------------------
                output = self._out_validator.validate(tool_name, raw_output)

                stability = 1.0 - (attempt * 0.1)

                return self._success_result(
                    tool_name,
                    tool.version,
                    output,
                    start,
                    stability,
                )

            except OutputValidationError as e:
                return self._failure_result(
                    tool_name,
                    tool.version,
                    f"Invalid output: {e}",
                    start,
                )

            except TimeoutError:
                error = f"Execution timed out after {timeout}s"

            except Exception as e:
                error = str(e)

            if attempt >= max_attempts - 1:
                return self._failure_result(
                    tool_name,
                    tool.version,
                    error,
                    start,
                )

        return self._failure_result(
            tool_name,
            tool.version,
            "Unknown execution state",
            start,
        )

    # ============================================================
    # RESULT BUILDERS
    # ============================================================

    def _success_result(
        self,
        tool_name: str,
        tool_version: str,
        output: Any,
        start_time: float,
        stability: float,
    ) -> ToolResult:

        return ToolResult(
            tool_name=tool_name,
            tool_version=tool_version,
            status="success",
            output=output,
            error=None,
            latency_ms=self._latency_ms(start_time),
            stability_signal=max(0.0, min(1.0, stability)),
        )

    def _failure_result(
        self,
        tool_name: str,
        tool_version: str,
        error: str,
        start_time: float,
    ) -> ToolResult:

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

        return ToolResult(
            tool_name=tool_name,
            tool_version="unknown",
            status="blocked",
            output=None,
            error=error,
            latency_ms=0,
            stability_signal=0.0,
        )

    @staticmethod
    def _latency_ms(start_time: float) -> int:
        return int((time.monotonic() - start_time) * 1000)

    # ------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------
    @property
    def registry(self) -> ToolRegistry:
        return self._registry
