from topomind.tools.schema import Tool


STATISTICS_TOOL = Tool(
    name="statistics",
    description=(
        "Perform statistical operations on numeric arrays. "
        "Supported operations:\n"
        "- mean: requires 'values'\n"
        "- std: requires 'values'\n"
        "- correlation: requires 'x' and 'y'"
    ),
    input_schema={
        "operation": "string",         # mean | std | correlation
        "values": "list[number]?",     # required for mean/std
        "x": "list[number]?",          # required for correlation
        "y": "list[number]?",          # required for correlation
    },
    output_schema={"result": "number"},
    connector_name="statistics",
    version="1.1.0",  # bumped version due to contract clarification
    tags=["builtin", "analytics", "statistics"],
)
