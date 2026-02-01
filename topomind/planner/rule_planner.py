from typing import List

from .interface import ReasoningEngine
from .plan_model import Plan, PlanStep
from ..models.tool_call import ToolCall
from ..tools.schema import Tool


class RuleBasedPlanner(ReasoningEngine):
    """
    Deterministic baseline planner.

    Used when no LLM planner is configured. Provides predictable,
    testable behavior and serves as a safe fallback.
    """

    def generate_plan(
        self,
        user_input: str,
        signals,
        tools: List[Tool],
    ) -> Plan:
        available_tools = {t.name for t in tools}

        # Fallback if echo tool not available
        if "echo" not in available_tools:
            return Plan(
                steps=[],
                goal="No valid tool available",
                meta={"reason": "echo tool missing"},
            )

        stable = signals.get("stable_entities", [])

        # Case 1: Reference persistent memory
        if stable:
            step = PlanStep(
                action=ToolCall(
                    tool_name="echo",
                    args={"text": f"Still talking about: {stable[0]}"},
                ),
                reasoning="Referenced stable entity from memory signals.",
                confidence=0.9,
            )
            return Plan(steps=[step], goal="Continue topic")

        # Case 2: Greeting
        if "hello" in user_input.lower():
            step = PlanStep(
                action=ToolCall(
                    tool_name="echo",
                    args={"text": "Hello from TopoMind Planner!"},
                ),
                reasoning="Greeting intent detected.",
                confidence=1.0,
            )
            return Plan(steps=[step], goal="Respond to greeting")

        # Default behavior
        step = PlanStep(
            action=ToolCall(
                tool_name="echo",
                args={"text": f"You said: {user_input}"},
            ),
            reasoning="Fallback echo behavior.",
            confidence=0.7,
        )

        return Plan(steps=[step], goal="Echo user input")
