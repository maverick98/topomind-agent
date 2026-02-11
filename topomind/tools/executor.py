from __future__ import annotations

import time
import requests
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
            return self._failure_result(tool_name, tool.version, f"Invalid arguments: {e}", start)

        # ------------------------------------------------------------
        # 🔥 Execution Model Routing (NEW)
        # ------------------------------------------------------------
        if tool.execution_model:
            try:
                args = self._generate_with_model(tool, args)
            except Exception as e:
                return self._failure_result(
                    tool_name,
                    tool.version,
                    f"Model generation failed: {e}",
                    start,
                )

        max_attempts = tool.max_retries + 1 if tool.retryable else 1
        timeout = tool.timeout_seconds

        # ------------------------------------------------------------
        # Execution Loop
        # ------------------------------------------------------------
        for attempt in range(max_attempts):
            try:
                raw_output = connector.execute(tool, args, timeout=timeout)

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
                return self._failure_result(tool_name, tool.version, f"Invalid output: {e}", start)

            except TimeoutError:
                error = f"Execution timed out after {timeout}s"

            except Exception as e:
                error = str(e)

            if attempt >= max_attempts - 1:
                return self._failure_result(tool_name, tool.version, error, start)

        return self._failure_result(tool_name, tool.version, "Unknown execution state", start)

    # ============================================================
    # 🔥 MODEL GENERATION LAYER
    # ============================================================

    def _generate_with_model(self, tool, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses tool.execution_model to generate Hawk DSL
        before invoking connector.
        """

        user_input = args.get("code", "")

        if not user_input:
            raise RuntimeError("Missing 'code' argument for model generation")

        # Merge tool prompt + user query
        full_prompt = f"{tool.prompt}\n\nUser request:\n{user_input}"

        payload = {
            "model": tool.execution_model,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": False,
            "options": {"temperature": 0},
        }

        logger.info("========== EXECUTION MODEL CALL ==========")
        logger.info(f"Model: {tool.execution_model}")
        logger.info("==========================================")

        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=180,
            proxies={"http": None, "https": None},
        )

        response.raise_for_status()

        data = response.json()
        generated_code = data.get("message", {}).get("content", "").strip()

        if not generated_code:
            raise RuntimeError("Model returned empty output")

        logger.info("========== GENERATED HAWK CODE ==========")
        logger.info(generated_code)
        logger.info("=========================================")

        return {"code": generated_code}

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
    # Accessors
    # ------------------------------------------------------------

    @property
    def registry(self) -> ToolRegistry:
        return self._registry
