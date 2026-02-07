import numpy as np
from typing import Dict, Any
from topomind.connectors.base import ExecutionConnector


class StatisticsConnector(ExecutionConnector):
    """
    Connector responsible for statistical operations.
    """

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        timeout: int,
    ) -> Any:

        operation = args["operation"]

        if operation == "mean":
            values = args["values"]
            return {"result": float(np.mean(values))}

        elif operation == "std":
            values = args["values"]
            return {"result": float(np.std(values))}

        elif operation == "correlation":
            x = args["x"]
            y = args["y"]

            if len(x) != len(y):
                raise ValueError("x and y must have same length")

            return {"result": float(np.corrcoef(x, y)[0, 1])}

        else:
            raise ValueError(f"Unsupported statistics operation: {operation}")
