from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ToolCall:
    """
    Represents a request from the Planner to execute a tool.

    This is an *intent packet* describing what action the agent
    wants to perform.
    """

    id: str
    tool_name: str
    arguments: Dict[str, Any]

    # Planner metadata
    confidence: float = 1.0
    reason: str = ""
