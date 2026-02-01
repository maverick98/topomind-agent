from dataclasses import dataclass, field
from typing import List, Optional

from ..models.tool_call import ToolCall
from ..models.tool_result import ToolResult
from ..planner.plan_model import Plan


@dataclass
class AgentState:
    """
    Represents the mutable runtime state of an agent session.

    This is NOT long-term memory. That belongs to the memory graph.
    AgentState tracks short-term execution context.
    """

    # ------------------------------------------------------------------
    # Conversation Context
    # ------------------------------------------------------------------

    turn_count: int = 0
    """
    Number of turns processed in this session.
    """

    last_user_input: Optional[str] = None
    """
    Most recent user input.
    """

    # ------------------------------------------------------------------
    # Planning Context
    # ------------------------------------------------------------------

    last_plan: Optional[Plan] = None
    """
    Last generated plan (useful for retries or stability checks).
    """

    # ------------------------------------------------------------------
    # Execution Context
    # ------------------------------------------------------------------

    last_tool_call: Optional[ToolCall] = None
    """
    Last tool invoked.
    """

    last_result: Optional[ToolResult] = None
    """
    Last tool execution result.
    """

    # ------------------------------------------------------------------
    # Short-Term History (bounded)
    # ------------------------------------------------------------------

    recent_results: List[ToolResult] = field(default_factory=list)
    """
    Short-term result history for immediate context.
    Not persisted long-term.
    """

    max_recent: int = 5
    """
    Maximum number of recent results to keep.
    """

    # ------------------------------------------------------------------
    # State Update Helpers
    # ------------------------------------------------------------------

    def new_turn(self, user_input: str) -> None:
        """Advance session turn and store latest user input."""
        self.turn_count += 1
        self.last_user_input = user_input

    def record_plan(self, plan: Plan) -> None:
        """Store last generated plan."""
        self.last_plan = plan

    def record_execution(self, tool_call: ToolCall, result: ToolResult) -> None:
        """
        Store execution details and maintain bounded history.
        """
        self.last_tool_call = tool_call
        self.last_result = result

        self.recent_results.append(result)
        if len(self.recent_results) > self.max_recent:
            self.recent_results.pop(0)
