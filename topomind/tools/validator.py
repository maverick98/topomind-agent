from __future__ import annotations

from typing import Dict, Any, Tuple

from .registry import ToolRegistry


class ArgumentValidationError(Exception):
    """Raised when tool arguments violate schema."""
    pass


class ArgumentValidator:
    """
    Validates tool arguments against tool input schemas.

    This prevents hallucinated parameters and enforces
    deterministic contracts before execution.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate arguments for a tool call.

        Returns sanitized args or raises ArgumentValidationError.
        """
        schema = self._registry.get_input_schema(tool_name)

        if not isinstance(args, dict):
            raise ArgumentValidationError("Arguments must be a dictionary.")

        self._check_required(schema, args)
        self._check_unknown(schema, args)
        self._check_types(schema, args)

        return args  # sanitized (future: coercion here)

    # ------------------------------------------------------------------
    # Validation Steps
    # ------------------------------------------------------------------

    def _check_required(self, schema: Dict[str, Any], args: Dict[str, Any]) -> None:
        missing = [k for k in schema if k not in args]
        if missing:
            raise ArgumentValidationError(f"Missing required arguments: {missing}")

    def _check_unknown(self, schema: Dict[str, Any], args: Dict[str, Any]) -> None:
        extra = [k for k in args if k not in schema]
        if extra:
            raise ArgumentValidationError(f"Unknown arguments: {extra}")

    def _check_types(self, schema: Dict[str, Any], args: Dict[str, Any]) -> None:
        for key, expected_type in schema.items():
            value = args[key]

            if not self._matches_type(expected_type, value):
                raise ArgumentValidationError(
                    f"Argument '{key}' expected type {expected_type}, got {type(value).__name__}"
                )

    # ------------------------------------------------------------------
    # Type Matching
    # ------------------------------------------------------------------

    def _matches_type(self, expected: Any, value: Any) -> bool:
        """
        Supports simple string type hints or Python types.
        """
        if isinstance(expected, str):
            return self._string_type_match(expected, value)

        if isinstance(expected, type):
            return isinstance(value, expected)

        return True  # Unknown schema type → don't block

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
            return True  # unknown schema spec → allow

        return isinstance(value, expected_type)
