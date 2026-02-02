from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator

from ..models.tool_call import ToolCall


@dataclass
class PlanStep:
    """
    A single reasoning step produced by the planner.

    A PlanStep couples a ToolCall with reasoning metadata that
    explains *why* the planner selected this action.

    This enables:
    • Traceable reasoning
    • Stability monitoring
    • Memory explanation storage
    """

    action: ToolCall
    """Tool action to execute."""

    reasoning: str
    """Planner explanation for the decision."""

    confidence: float = 1.0
    """
    Planner confidence score in range [0.0, 1.0].

    Values outside the range are automatically clamped.
    """

    def __post_init__(self):
        # Safety normalization
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class Plan:
    """
    Structured output of a ReasoningEngine.

    A Plan represents the planner’s intention over one or more steps.

    Current system executes only the first step, but the structure
    supports future multi-step and branching plans.

    Architectural role:
        Planner output → Agent execution → Memory trace
    """

    steps: List[PlanStep] = field(default_factory=list)
    """Ordered sequence of reasoning steps."""

    goal: Optional[str] = None
    """High-level objective guiding this plan."""

    meta: Dict[str, Any] = field(default_factory=dict)
    """
    Optional planner metadata.

    Examples:
    • model name
    • token usage
    • planner type
    """

    # ------------------------------------------------------------------
    # Convenience Accessors
    # ------------------------------------------------------------------

    @property
    def first_step(self) -> PlanStep:
        """
        Return the first reasoning step.

        Raises
        ------
        ValueError
            If plan contains no steps.
        """
        if not self.steps:
            raise ValueError("Cannot access first_step of empty plan.")
        return self.steps[0]

    def is_empty(self) -> bool:
        """Return True if planner produced no actions."""
        return len(self.steps) == 0

    def size(self) -> int:
        """Return number of steps in the plan."""
        return len(self.steps)

    # ------------------------------------------------------------------
    # Iteration Support (future multi-step execution)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[PlanStep]:
        """Allow iteration over plan steps."""
        return iter(self.steps)
