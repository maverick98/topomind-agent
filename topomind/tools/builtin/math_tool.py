from topomind.tools.schema import Tool


MATH_TOOL = Tool(
    name="calculate",
    description=(
        "Perform arithmetic calculations safely using Python syntax. "
        "Use operators + - * / ** %. "
        "Use math.sqrt(), math.sin(), etc. "
        "Do NOT wrap expression in quotes."
    ),
    input_schema={
        "expression": "string"
    },
    output_schema={"result": "string"},
    connector_name="math",
    version="1.1.0",
    tags=["builtin", "math"],
)
