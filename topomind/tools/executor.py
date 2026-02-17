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
            validated_args = self._arg_validator.validate(tool_name, args)
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
                working_args = dict(validated_args)

                # ============================================================
                # CASE 1: LLM-ASSISTED TOOL
                # ============================================================
                if tool.execution_model:

                    llm_connector = self._connectors.get("llm")

                    if not tool.prompt:
                        raise RuntimeError(
                            f"Tool '{tool.name}' has execution_model but no prompt defined"
                        )

                    logger.info(
                        f"[EXECUTION MODEL] Using model: {tool.execution_model}"
                    )

                    # 🔹 Structured execution
                    generated_output = llm_connector.execute(
                        system_prompt=tool.prompt,
                        user_args=working_args,
                        model=tool.execution_model,
                        timeout=timeout,
                    )

                    raw_output = self._normalize_output(
                        tool,
                        generated_output
                    )

                # ============================================================
                # CASE 2: PURE DETERMINISTIC TOOL
                # ============================================================
                else:
                    raw_output = connector.execute(
                        tool.name,
                        working_args,
                        timeout=timeout,
                    )

                # ------------------------------------------------------------
                # Output Validation
                # ------------------------------------------------------------
                output = self._out_validator.validate(tool_name, raw_output)

                stability = 1.0 - (attempt / max_attempts)

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
    # Output Normalization
    # ============================================================

    def _normalize_output(self, tool, generated_output):

        # LLM returned structured dict
        if isinstance(generated_output, dict):
            return generated_output

        # LLM returned raw string → wrap according to schema
        if isinstance(generated_output, str):

            schema_fields = list(tool.output_schema.keys())

            if len(schema_fields) != 1:
                raise RuntimeError(
                    f"Tool '{tool.name}' expects structured output "
                    f"{schema_fields}, but received raw string."
                )

            return {schema_fields[0]: generated_output}

        raise RuntimeError(
            f"Unsupported output type from connector for tool '{tool.name}'"
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

    @property
    def registry(self) -> ToolRegistry:
        return self._registry
