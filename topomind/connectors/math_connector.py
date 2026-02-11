from typing import Dict, Any
from topomind.connectors.base import ExecutionConnector
from .safe_math import SafeExpressionEvaluator


class MathConnector(ExecutionConnector):
    """
    Deterministic arithmetic connector using safe AST evaluation.
    """

    def __init__(self):
        self._evaluator = SafeExpressionEvaluator()

    def execute(self, tool, args: Dict[str, Any], timeout: int)-> Any:

        expression = args["expression"]

        try:
            result = self._evaluator.evaluate(expression)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

        return {"result": str(result)}
