from topomind.tools.schema import Tool


STATISTICS_TOOL = Tool(
    name="statistics",
    description="Enterprise statistical operations.",
    input_schema={
        "operation": "string",
        "values": "list[number]?",
        "x": "list[number]?",
        "y": "list[number]?",
        "lag": "int?",
    },
    output_schema={"result": "any"},
    connector_name="statistics",
    version="2.0.0",
    tags=["builtin", "analytics", "statistics"],
)
