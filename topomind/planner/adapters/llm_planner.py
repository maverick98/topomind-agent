import json
import uuid
import logging
import time
from typing import List

from ..interface import ReasoningEngine
from ..plan_model import Plan, PlanStep
from ..prompt_builder import PlannerPromptBuilder
from ...models.tool_call import ToolCall
from ...tools.schema import Tool
from ...agent.llm.llm_client import LLMClient
from .utils import extract_first_json

logger = logging.getLogger(__name__)


class LLMPlanner(ReasoningEngine):

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.prompt_builder = PlannerPromptBuilder()

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:

        signals = signals or {}
        tools = sorted(tools, key=lambda t: t.name)

        prompt = self.prompt_builder.build(
            user_input=user_input,
            signals=signals,
            tools=tools,
        )

        strict_mode_enabled = any(getattr(t, "strict", False) for t in tools)

        start_time = time.time()

        try:
            raw_text = self.llm.chat(prompt, strict=strict_mode_enabled)
            latency = time.time() - start_time

            try:
                result = json.loads(raw_text)
            except Exception:
                extracted = extract_first_json(raw_text)
                if not extracted:
                    raise ValueError("No JSON object found in LLM output")
                result = json.loads(extracted)

            steps_json = result.get("steps", [])
            confidence = float(result.get("confidence", 0.7))

            valid_tool_names = {t.name for t in tools}
            plan_steps = []

            for s in steps_json:
                tool_name = s.get("tool")
                args = s.get("args", {})

                if tool_name not in valid_tool_names:
                    raise ValueError(f"Invalid tool selected: {tool_name}")

                plan_steps.append(
                    PlanStep(
                        action=ToolCall(
                            id=str(uuid.uuid4()),
                            tool_name=tool_name,
                            arguments=args,
                        ),
                        reasoning="LLM planner decision",
                        confidence=confidence,
                    )
                )

            return Plan(
                steps=plan_steps,
                goal="LLM planning",
                meta={
                    "planner": self.__class__.__name__,
                    "model": getattr(self.llm, "model", None),
                    "latency_seconds": latency,
                },
            )

        except Exception as e:
            return self._fallback_plan(tools, str(e))
        
    def _fallback_plan(self, tools, error_message: str, user_input: str = "") -> Plan:
        """
        Fallback plan when LLM returns invalid JSON.
        Default behavior: call compileHawk with user query.
        """

        logger.warning(f"[PLANNER FALLBACK] Triggered due to: {error_message}")

        if not tools:
            return Plan(
                steps=[],
                goal="Fallback planning",
                meta={"error": error_message},
            )

        # Prefer compileHawk if available
        compile_tool = next((t for t in tools if t.name == "compileHawk"), None)

        if compile_tool:
            tool_name = compile_tool.name
            arguments = {"query": user_input}
        else:
            first_tool = sorted(tools, key=lambda t: t.name)[0]
            tool_name = first_tool.name
            arguments = {}

        fallback_step = PlanStep(
            action=ToolCall(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                arguments=arguments,
            ),
            reasoning="Fallback planner decision",
            confidence=0.0,
        )

        return Plan(
            steps=[fallback_step],
            goal="Fallback planning",
            meta={
                "planner": self.__class__.__name__,
                "fallback": True,
                "error": error_message,
            },
        )
