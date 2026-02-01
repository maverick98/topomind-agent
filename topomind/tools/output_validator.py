from __future__ import annotations

from typing import Dict, Any

from .registry import ToolRegistry


class OutputValidationError(Exception):
    """Raised when tool output violates declared schema."""
    pass


class OutputValidator:
    """
    Validates tool outputs against declared output schemas.

    This protects memory, planner context, and reasoning layers
    from malformed or unexpected tool responses.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, tool_name: str, output: Any) -> Any:
        """
        Validate tool output. Returns output if valid.

        Raises OutputValidationError if schema is violated.
        """
        schema = self._registry.get_output_schema(tool_name)

        if not isinstance(output, dict):
            raise OutputValidationError("Tool output must be a dictionary.")

        self._check_required(schema, output)
        self._check_unknown(schema, output)
        self._check_types(schema, output)

        return output

    # ------------------------------------------------------------------
    # Validation Steps
    # ------------------------------------------------------------------

    def _check_required(self, schema: Dict[str, Any], output: Dict[str, Any]) -> None:
        missing = [k for k in schema if k not in output]
        if missing:
            raise OutputValidationError(f"Missing output fields: {missing}")

    def _check_unknown(self, schema: Dict[str, Any], output: Dict[str, Any]) -> None:
        extra = [k for k in output if k not in schema]
        if extra:
            raise OutputValidationError(f"Unexpected output fields: {extra}")

    def _check_types(self, schema: Dict[str, Any], output: Dict[str, Any]) -> None:
        for key, expected_type in schema.items():
            value = output[key]

            if not self._matches_type(expected_type, value):
                raise OutputValidationError(
                    f"Output field '{key}' expected type {expected_type}, got {type(value).__name__}"
                )

    # ------------------------------------------------------------------
    # Type Matching
    # ------------------------------------------------------------------

    def _matches_type(self, expected: Any, value: Any) -> bool:
        if isinstance(expected, str):
            return self._string_type_match(expected, value)

        if isinstance(expected, type):
            return isinstance(value, expected)

        return True

    def _string_type_match(self, expected: str, value: Any) -> bool:
        mapping = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "dict": dict,
            "list": list,
        }

        expected_type = mapping.get(expected.lower())
        if expected_type is None:
            return True

        return isinstance(value, expected_type)
