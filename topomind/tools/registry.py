from typing import Dict
from .schema import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Tool:
        return self._tools[tool_name]

    def list_tools(self):
        return list(self._tools.values())
