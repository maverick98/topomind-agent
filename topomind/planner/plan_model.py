from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..models.tool_call import ToolCall


@dataclass
class PlanStep:
    """
    A single reasoning step in a plan.

    Wraps a ToolCall with planner-level reasoning metadata.
    """

    action: ToolCall
    """
    The tool action to execute.
    """

    reasoning: str
    """
    Planner explanation for why this action was chosen.
    Used by stability analysis and memory trace logging.
    """

    confidence: float = 1.0
    """
    Planner confidence in this step (0.0–1.0).
    Can later be used by stability monitoring.
    """


@dataclass
class Plan:
    """
    Structured output of a ReasoningEngine.

    A plan is an ordered sequence of reasoning steps.
    The current system executes the first step, but the structure
    supports future multi-step planning.
    """

    steps: List[PlanStep] = field(default_factory=list)
    """
    Ordered reasoning steps to execute.
    """

    goal: Optional[str] = None
    """
    High-level objective the planner is pursuing.
    Useful for long-term reasoning and memory tracking.
    """

    meta: Dict[str, Any] = field(default_factory=dict)
    """
    Additional planner metadata (model name, tokens used, etc.).
    Not required for execution.
    """

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    @property
    def first_step(self) -> PlanStep:
        """
        Return the first step of the plan.

        Most planners currently produce single-step plans.
        """
        return self.steps[0]

    def is_empty(self) -> bool:
        """Check whether the planner produced no actions."""
        return len(self.steps) == 0
