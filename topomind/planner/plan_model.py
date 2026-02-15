from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator

from ..models.tool_call import ToolCall


# ============================================================
# PlanStep
# ============================================================

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
        # Normalize and clamp confidence safely
        try:
            self.confidence = float(self.confidence)
        except Exception:
            self.confidence = 0.0

        self.confidence = max(0.0, min(1.0, self.confidence))


# ============================================================
# Plan
# ============================================================

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
    • backend used
    • retry count
    """

    # ---------------------------------------------------------
    # Convenience Accessors
    # ---------------------------------------------------------

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

    @property
    def first_step_or_none(self) -> Optional[PlanStep]:
        """Safely return first step or None if plan is empty."""
        return self.steps[0] if self.steps else None

    @property
    def confidence(self) -> float:
        """
        Aggregate plan confidence.

        Current logic:
        - If empty → 0.0
        - Single-step → that step's confidence
        - Multi-step → minimum confidence (conservative)
        """
        if not self.steps:
            return 0.0

        return min(step.confidence for step in self.steps)

    # ---------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------

    def is_empty(self) -> bool:
        """Return True if planner produced no actions."""
        return len(self.steps) == 0

    def size(self) -> int:
        """Return number of steps in the plan."""
        return len(self.steps)

    # ---------------------------------------------------------
    # Iteration Support (future multi-step execution)
    # ---------------------------------------------------------

    def __iter__(self) -> Iterator[PlanStep]:
        """Allow iteration over plan steps."""
        return iter(self.steps)

    # ---------------------------------------------------------
    # Debug / Observability Helpers
    # ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert plan to serializable dictionary.
        Useful for logging, tracing, memory storage.
        """
        return {
            "goal": self.goal,
            "confidence": self.confidence,
            "steps": [
                {
                    "tool": step.action.tool_name,
                    "arguments": step.action.arguments,
                    "reasoning": step.reasoning,
                    "confidence": step.confidence,
                }
                for step in self.steps
            ],
            "meta": self.meta,
        }
