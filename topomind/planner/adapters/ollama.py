import json
import uuid
import requests
import logging
import time
from typing import List
import re

from ..interface import ReasoningEngine
from ..plan_model import Plan, PlanStep
from ..prompt_builder import PlannerPromptBuilder
from ...models.tool_call import ToolCall
from ...tools.schema import Tool

logger = logging.getLogger(__name__)


class OllamaPlanner(ReasoningEngine):

    def __init__(self, model: str = "phi3:mini"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"
        self.prompt_builder = PlannerPromptBuilder()

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:

        tools = sorted(tools, key=lambda t: t.name)

        # ==============================
        # DEBUG 1 — TOOL MANIFEST
        # ==============================
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

        # Hard constraint
        valid_tools = [t.name for t in tools]

        prompt += "\n\nYou MUST choose exactly one of:\n"
        for t in valid_tools:
            prompt += f"- {t}\n"
        prompt += "\nReturn STRICT JSON.\n"
        prompt += "Double quotes only.\n"
        prompt += "No single quotes.\n"

        # ==============================
        # DEBUG 2 — PROMPT TRACE
        # ==============================
        logger.info("========== PROMPT SENT TO LLM ==========")
        logger.info(prompt[:2000])  # prevent explosion
        logger.info("========================================")

        strict_mode_enabled = any(getattr(t, "strict", False) for t in tools)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 96,
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
            proxies={"http": None, "https": None},
            timeout=120
        )

        elapsed = time.time() - start_time
        logger.info(f"[PLANNER] LLM latency: {elapsed:.2f}s")

        response_json = response.json()

        # ==============================
        # DEBUG 3 — RAW LLM OUTPUT
        # ==============================
        raw_text = response_json.get("message", {}).get("content", "")
        logger.info("========== RAW LLM OUTPUT ==========")
        logger.info(raw_text)
        logger.info("====================================")

        try:
            cleaned = raw_text.strip()

            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
                cleaned = cleaned.rstrip("`").strip()

            cleaned = cleaned.replace("'", '"')
            cleaned = cleaned.replace('"""', '"')

            # Extract first JSON block
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            # ==============================
            # DEBUG 4 — CLEANED JSON
            # ==============================
            logger.info("========== CLEANED JSON ==========")
            logger.info(cleaned)
            logger.info("==================================")

            result = json.loads(cleaned)

            tool_name = result.get("tool")
            args = result.get("args", {})

            logger.info(f"[PARSED TOOL] {tool_name}")
            logger.info(f"[PARSED ARGS] {args}")
            logger.info(f"[CONFIDENCE] {result.get('confidence')}")

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

            # fallback
            fallback_tool = tools[0].name if tools else "none"

            logger.warning(f"[FALLBACK] Using {fallback_tool}")

            step = PlanStep(
                action=ToolCall(
                    id=str(uuid.uuid4()),
                    tool_name=fallback_tool,
                    arguments={},
                ),
                reasoning="Planner failed — fallback",
                confidence=0.0,
            )

            return Plan(steps=[step], goal="Planner fallback")
