# topomind/tools/builtin/reason_tool.py

from topomind.tools.schema import Tool

ReasonTool = Tool(
    name="reason",
    description="Use LLM to answer conceptual or knowledge questions",
    input_schema={"question": "string"},
    output_schema={"answer": "string"},
    connector_name="llm",   # connector that talks to Ollama
)
