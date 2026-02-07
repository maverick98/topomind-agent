from __future__ import annotations

from typing import Dict, Any

from .registry import ToolRegistry


class ArgumentValidationError(Exception):
    """Raised when tool arguments violate schema."""
    pass


class ArgumentValidator:
    """
    Validates tool arguments against tool input schemas.
    Supports:
    - Optional fields via '?'
    - list[number] style hints
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:

        schema = self._registry.get_input_schema(tool_name)

        if not isinstance(args, dict):
            raise ArgumentValidationError("Arguments must be a dictionary.")

        self._check_required(schema, args)
        self._check_unknown(schema, args)
        self._check_types(schema, args)

        return args

    # ------------------------------------------------------------------
    # Validation Steps
    # ------------------------------------------------------------------

    def _check_required(self, schema: Dict[str, Any], args: Dict[str, Any]) -> None:
        missing = []

        for key, expected_type in schema.items():
            is_optional = isinstance(expected_type, str) and expected_type.endswith("?")

            if not is_optional and key not in args:
                missing.append(key)

        if missing:
            raise ArgumentValidationError(f"Missing required arguments: {missing}")

    def _check_unknown(self, schema: Dict[str, Any], args: Dict[str, Any]) -> None:
        extra = [k for k in args if k not in schema]
        if extra:
            raise ArgumentValidationError(f"Unknown arguments: {extra}")

    def _check_types(self, schema: Dict[str, Any], args: Dict[str, Any]) -> None:

        for key, expected_type in schema.items():

            if key not in args:
                continue  # optional and not present

            clean_expected = (
                expected_type.rstrip("?")
                if isinstance(expected_type, str)
                else expected_type
            )

            value = args[key]

            if not self._matches_type(clean_expected, value):
                raise ArgumentValidationError(
                    f"Argument '{key}' expected type {clean_expected}, got {type(value).__name__}"
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

        expected = expected.lower()

        # -------------------------
        # Simple types
        # -------------------------

        mapping = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "dict": dict,
            "list": list,
        }

        if expected in mapping:
            return isinstance(value, mapping[expected])

        # -------------------------
        # list[number]
        # -------------------------

        if expected == "list[number]":
            if not isinstance(value, list):
                return False
            return all(isinstance(v, (int, float)) for v in value)

        if expected == "list[string]":
            if not isinstance(value, list):
                return False
            return all(isinstance(v, str) for v in value)

        return True  # Unknown spec → allow
