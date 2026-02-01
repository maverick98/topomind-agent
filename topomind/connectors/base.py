from abc import ABC, abstractmethod
from typing import Dict, Any


class ExecutionConnector(ABC):
    @abstractmethod
    def execute(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool call in an external deterministic system.
        """
        pass
class FakeConnector(ExecutionConnector):
    def execute(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name == "echo":
            return args.get("text", "")
        return f"Tool {tool_name} executed with args {args}"
