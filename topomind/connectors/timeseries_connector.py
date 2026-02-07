import numpy as np
import pandas as pd
from typing import Dict, Any
from topomind.connectors.base import ExecutionConnector


class TimeSeriesConnector(ExecutionConnector):
    """
    Connector responsible for time-series transformations.
    """

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        timeout: int,
    ) -> Any:

        operation = args["operation"]
        values = args["values"]

        if operation == "moving_average":
            window = args["window"]
            result = (
                pd.Series(values)
                .rolling(window)
                .mean()
                .tolist()
            )
            return {"result": result}

        elif operation == "cumulative_sum":
            return {"result": list(np.cumsum(values))}

        else:
            raise ValueError(f"Unsupported timeseries operation: {operation}")
