from topomind.tools.schema import Tool


TIMESERIES_TOOL = Tool(
    name="timeseries",
    description=(
        "Perform time-series transformations. "
        "Supported operations:\n"
        "- moving_average: requires 'values' and 'window'\n"
        "- cumulative_sum: requires 'values'"
    ),
    input_schema={
        "operation": "string",         # moving_average | cumulative_sum
        "values": "list[number]",
        "window": "int?",              # required only for moving_average
    },
    output_schema={"result": "list[number]"},
    connector_name="timeseries",
    version="1.1.0",
    tags=["builtin", "analytics", "timeseries"],
)
