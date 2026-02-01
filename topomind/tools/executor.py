from typing import Dict, Any
from .registry import ToolRegistry
from ..connectors.manager import ConnectorManager


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, connectors: ConnectorManager):
        self.registry = registry
        self.connectors = connectors

    def execute(self, tool_name: str, args: Dict[str, Any]):
        tool = self.registry.get(tool_name)
        connector = self.connectors.get(tool.connector_name)
        return connector.execute(tool_name, args)
