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

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:

        print(f"model inside OllamaPlanner is {self.model}")

        tools = sorted(tools, key=lambda t: t.name)

        logger.info("========== PLANNER DEBUG ==========")
        logger.info(f"[INPUT] {user_input}")
        logger.info(f"[SIGNALS] {signals}")
        logger.info(f"[AVAILABLE TOOLS] {[t.name for t in tools]}")
        logger.info("====================================")

        # 🚫 REMOVED SINGLE TOOL FAST PATH
        # ALWAYS USE LLM

        prompt = self.prompt_builder.build(
            user_input=user_input,
            signals=signals,
            tools=tools,
        )

        valid_tools = [t.name for t in tools]

        prompt += "\n\nYou MUST choose exactly one of:\n"
        for t in valid_tools:
            prompt += f"- {t}\n"
        prompt += "\nReturn STRICT JSON.\n"
        prompt += "Double quotes only.\n"
        prompt += "No markdown.\n"
        prompt += "No explanation outside JSON.\n"

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
                "num_predict": 128,
            }
        }

        if strict_mode_enabled:
            payload["options"]["temperature"] = 0
            logger.info("[PLANNER] Strict mode ON")

        logger.info(f"[PLANNER] Model: {self.model}")

        start_time = time.time()

        response = requests.post(
            self.url,
            json=payload,
            proxies={"http": None, "https": None}
        )

        elapsed = time.time() - start_time
        logger.info(f"[PLANNER] LLM latency: {elapsed:.2f}s")
        logger.info(f"[OLLAMA STATUS] {response.status_code}")

        if response.status_code != 200:
            logger.error("[OLLAMA ERROR] Non-200 response")
            return self._fallback_plan(tools, "Ollama HTTP failure")

        response_json = response.json()
        raw_text = response_json.get("message", {}).get("content", "")

        logger.info("========== RAW LLM OUTPUT ==========")
        logger.info(raw_text)
        logger.info("====================================")

        try:
            extracted_json = extract_first_json(raw_text)

            if not extracted_json:
                raise ValueError("No JSON object found in LLM output")

            logger.info("========== EXTRACTED JSON ==========")
            logger.info(extracted_json)
            logger.info("====================================")

            result = json.loads(extracted_json)

            tool_name = result.get("tool")
            args = result.get("args", {})

            valid_tool_names = {t.name for t in tools}
            if tool_name not in valid_tool_names:
                raise ValueError(
                    f"Invalid tool selected: {tool_name}. "
                    f"Valid tools: {valid_tool_names}"
                )

            if not isinstance(args, dict):
                raise ValueError("Args must be dictionary")

            step = PlanStep(
                action=ToolCall(
                    id=str(uuid.uuid4()),
                    tool_name=tool_name,
                    arguments=args,
                ),
                reasoning=result.get("reasoning", "LLM decision"),
                confidence=float(result.get("confidence", 0.7)),
            )

            logger.info("========== PLAN CREATED ==========")
            logger.info(f"Tool: {tool_name}")
            logger.info(f"Confidence: {step.confidence}")
            logger.info("==================================")

            return Plan(steps=[step], goal="LLM-driven planning")

        except Exception as e:
            logger.error("========== PLANNER FAILURE ==========")
            logger.error(f"Error: {e}")
            logger.error("=====================================")

            return self._fallback_plan(tools, str(e))

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
