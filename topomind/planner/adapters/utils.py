import json
import uuid
import logging
import time
from typing import List, Optional

from ..interface import ReasoningEngine
from ..plan_model import Plan, PlanStep
from ..prompt_builder import PlannerPromptBuilder
from ...models.tool_call import ToolCall
from ...tools.schema import Tool
from ...agent.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Safe JSON Extraction
# ------------------------------------------------------------

def extract_first_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i + 1]

    return None


# ============================================================
# LLM Planner
# ============================================================

class LLMPlanner(ReasoningEngine):

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.prompt_builder = PlannerPromptBuilder()

    # ============================================================
    # MAIN PLANNING
    # ============================================================

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:

        signals = signals or {}
        tools = sorted(tools, key=lambda t: t.name)

        logger.info("========== LLM PLANNER ==========")
        logger.info(f"[INPUT] {user_input}")
        logger.info(f"[TOOLS] {[t.name for t in tools]}")
        logger.info("=================================")

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

            logger.info(f"[LLM LATENCY] {latency:.2f}s")
            logger.info("========== RAW OUTPUT ==========")
            logger.info(raw_text)
            logger.info("================================")

            # --------------------------------------------------
            # Try direct JSON parse first
            # --------------------------------------------------
            try:
                result = json.loads(raw_text)
            except Exception:
                extracted = extract_first_json(raw_text)
                if not extracted:
                    raise ValueError("No JSON object found in LLM output")
                result = json.loads(extracted)

            steps_json = result.get("steps", [])
            confidence = float(result.get("confidence", 0.7))

            if not isinstance(steps_json, list) or not steps_json:
                raise ValueError("Planner returned no steps")

            valid_tool_names = {t.name for t in tools}
            plan_steps = []

            for s in steps_json:

                tool_name = s.get("tool")
                args = s.get("args", {})

                if tool_name not in valid_tool_names:
                    raise ValueError(
                        f"Invalid tool selected: {tool_name}"
                    )

                if not isinstance(args, dict):
                    raise ValueError("Tool args must be JSON object")

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

            logger.info(f"[PLAN] Selected tool: {plan_steps[0].action.tool_name}")

            return Plan(
                steps=plan_steps,
                goal="LLM planning",
                meta={
                    "planner": self.name,
                    "model": getattr(self.llm, "model", None),
                    "latency_seconds": latency,
                },
            )

        except Exception as e:
            logger.error("========== PLANNER FAILURE ==========")
            logger.error(str(e))
            logger.error("=====================================")

            return self._fallback_plan(tools, str(e))

    # ============================================================
    # FALLBACK
    # ============================================================

    def _fallback_plan(self, tools: List[Tool], reason: str) -> Plan:

        logger.warning(f"[FALLBACK REASON] {reason}")

        # Try safe echo fallback if available
        for t in tools:
            if t.name == "echo":
                return Plan(
                    steps=[
                        PlanStep(
                            action=ToolCall(
                                id=str(uuid.uuid4()),
                                tool_name="echo",
                                arguments={"text": "Planner fallback triggered."},
                            ),
                            reasoning="LLM planner failure fallback",
                            confidence=0.0,
                        )
                    ],
                    goal="Planner fallback",
                    meta={"fallback": True, "reason": reason},
                )

        # If no safe tool available → return empty plan
        return Plan(
            steps=[],
            goal="Planner fallback",
            meta={"fallback": True, "reason": reason},
        )
