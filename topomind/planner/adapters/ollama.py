import json
import uuid
import requests
import logging
import time
from typing import List, Optional

from ..interface import ReasoningEngine
from ..plan_model import Plan, PlanStep
from ..prompt_builder import PlannerPromptBuilder
from ...models.tool_call import ToolCall
from ...tools.schema import Tool

logger = logging.getLogger(__name__)


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


class OllamaPlanner(ReasoningEngine):

    def __init__(self, model: str = "phi3:mini"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"
        self.prompt_builder = PlannerPromptBuilder()

    # ============================================================
    # MAIN PLANNING
    # ============================================================

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:

        print(f"model inside OllamaPlanner is {self.model}")

        tools = sorted(tools, key=lambda t: t.name)

        logger.info("========== PLANNER DEBUG ==========")
        logger.info(f"[INPUT] {user_input}")
        logger.info(f"[SIGNALS] {signals}")
        logger.info(f"[AVAILABLE TOOLS] {[t.name for t in tools]}")
        logger.info("====================================")

        prompt = self.prompt_builder.build(
            user_input=user_input,
            signals=signals,
            tools=tools,
        )

        prompt += """

You MUST return STRICT JSON with this structure:

{
  "steps": [
    {
      "tool": "tool_name",
      "args": { }
    }
  ],
  "confidence": 0.0-1.0
}

Rules:
- You MUST return EXACTLY ONE step.
- Multi-step planning is NOT allowed.
- The planner NEVER fabricates intermediate outputs.
- The planner NEVER computes results.
- The planner ONLY selects the highest-level tool needed.
- Tools must be selected only from the available list.
- Arguments must be valid JSON objects.
- No markdown.
- No explanation outside JSON.
"""

        logger.info(f"[PROMPT LENGTH] {len(prompt)} chars")
        logger.info("========== PROMPT SENT TO LLM ==========")
        logger.info(prompt[:2000])
        logger.info("========================================")

        strict_mode_enabled = any(getattr(t, "strict", False) for t in tools)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 256,
            }
        }

        if strict_mode_enabled:
            payload["options"]["temperature"] = 0
            logger.info("[PLANNER] Strict mode ON")

        logger.info(f"[PLANNER] Model: {self.model}")

        start_time = time.time()

        try:
            response = requests.post(
                self.url,
                json=payload,
                proxies={"http": None, "https": None}
            )

            elapsed = time.time() - start_time
            logger.info(f"[PLANNER] LLM latency: {elapsed:.2f}s")
            logger.info(f"[OLLAMA STATUS] {response.status_code}")

            if response.status_code != 200:
                raise RuntimeError("Non-200 response from Ollama")

            response_json = response.json()
            raw_text = response_json.get("message", {}).get("content", "")

            logger.info("========== RAW LLM OUTPUT ==========")
            logger.info(raw_text)
            logger.info("====================================")

            extracted_json = extract_first_json(raw_text)

            if not extracted_json:
                raise ValueError("No JSON object found in LLM output")

            logger.info("========== EXTRACTED JSON ==========")
            logger.info(extracted_json)
            logger.info("====================================")

            result = json.loads(extracted_json)

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
                        f"Invalid tool selected: {tool_name}. "
                        f"Valid tools: {valid_tool_names}"
                    )

                if not isinstance(args, dict):
                    raise ValueError("Args must be dictionary")

                plan_steps.append(
                    PlanStep(
                        action=ToolCall(
                            id=str(uuid.uuid4()),
                            tool_name=tool_name,
                            arguments=args,
                        ),
                        reasoning="LLM multi-step decision",
                        confidence=confidence,
                    )
                )

            logger.info("========== MULTI-STEP PLAN CREATED ==========")
            logger.info(f"Steps: {[s.action.tool_name for s in plan_steps]}")
            logger.info("============================================")

            return Plan(steps=plan_steps, goal="LLM multi-step planning")

        except Exception as e:
            logger.error("========== PLANNER FAILURE ==========")
            logger.error(f"Error: {e}")
            logger.error("=====================================")

            return self._fallback_plan(tools, str(e))

    # ============================================================
    # FALLBACK
    # ============================================================

    def _fallback_plan(self, tools: List[Tool], reason: str) -> Plan:

        fallback_tool = tools[0].name if tools else "none"

        logger.warning(f"[FALLBACK] Using {fallback_tool}")
        logger.warning(f"[REASON] {reason}")

        step = PlanStep(
            action=ToolCall(
                id=str(uuid.uuid4()),
                tool_name=fallback_tool,
                arguments={},
            ),
            reasoning="Planner failed — deterministic fallback",
            confidence=0.0,
        )

        return Plan(steps=[step], goal="Planner fallback")
