from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..tools.schema import Tool


class ReasoningEngine(ABC):
    @abstractmethod
    def generate_plan(
        self,
        user_input: str,
        signals: Dict[str, Any],
        tools: List[Tool]
    ) -> Dict[str, Any]:
        """
        Convert natural language input into a structured plan.
        Tools list tells planner what it can call.
        """
        pass
