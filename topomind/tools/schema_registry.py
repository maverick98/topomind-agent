from typing import Dict, Tuple
from .schema import Tool


class SchemaRegistry:
    """
    Stores historical versions of tool schemas.
    """

    def __init__(self):
        self._schemas: Dict[Tuple[str, str], Tool] = {}

    def register(self, tool: Tool):
        key = (tool.name, tool.version)
        self._schemas[key] = tool

    def get(self, tool_name: str, version: str) -> Tool:
        return self._schemas[(tool_name, version)]
