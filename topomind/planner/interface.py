from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from ..tools.schema import Tool
from .plan_model import Plan


class ReasoningEngine(ABC):
    """
    Abstract reasoning interface.

    A ReasoningEngine converts user input + system signals +
    available tool capabilities into a structured execution plan.

    Implementations may be:
    - Rule-based (deterministic)
    - LLM-based (Ollama/Groq/etc.)
    - Hybrid planners
    """

    @property
    def name(self) -> str:
        """
        Return planner identity.

        Useful for:
        • logging
        • telemetry
        • memory traces
        • debugging
        """
        return self.__class__.__name__

    @abstractmethod
    def generate_plan(
        self,
        user_input: str,
        signals: Optional[Dict[str, Any]],
        tools: List[Tool],
    ) -> Plan:
        """
        Generate a structured plan.

        Parameters
        ----------
        user_input : str
            Raw user request.

        signals : Dict[str, Any] | None
            System-level signals and memory-derived context.
            Example:
                - stable_entities
                - constraints
                - system health

        tools : List[Tool]
            Tools available for planning. Defines the action space.

        Returns
        -------
        Plan
            Structured reasoning output describing which tool(s)
            to call and why.
        """
        raise NotImplementedError
