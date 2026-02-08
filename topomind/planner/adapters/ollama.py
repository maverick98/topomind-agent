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
    """
    LLM-based planner that decides WHICH TOOL to use.
    It NEVER produces final answers.
    """

    def __init__(self, model: str = "phi3:mini"):   # switched model
        self.model = model
        self.url = "http://localhost:11434/api/chat"
        self.prompt_builder = PlannerPromptBuilder()

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:

        tools = sorted(tools, key=lambda t: t.name)

        prompt = self.prompt_builder.build(
            user_input=user_input,
            signals=signals,
            tools=tools,
        )

        strict_mode_enabled = any(getattr(t, "strict", False) for t in tools)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 256,   # planner does not need 512
            }
        }

        if strict_mode_enabled:
            logger.info("[PLANNER] Strict tool detected. Forcing temperature=0.")
            payload["options"]["temperature"] = 0

        logger.info("[PLANNER] Sending request to Ollama...")
        logger.info("[PLANNER DEBUG] Model: %s", payload["model"])
        logger.info(
            "[PLANNER DEBUG] Temperature: %s",
            payload.get("options", {}).get("temperature", "default")
        )

        start_time = time.time()

        try:
            response = requests.post(
                self.url,
                json=payload,
                proxies={"http": None, "https": None},
                timeout=180   # realistic for CPU
            )
        except Exception as e:
            logger.error(f"[PLANNER] Exception while contacting Ollama: {e}")
            raise

        elapsed = time.time() - start_time
        logger.info(f"[PLANNER] Ollama responded in {elapsed:.2f}s")

        response_json = response.json()
        text = response_json.get("message", {}).get("content", "").strip()

        try:
            #  non-greedy JSON extraction
            cleaned = text.strip()

            # Remove markdown fences if present
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
                cleaned = cleaned.rstrip("`").strip()

            result = json.loads(cleaned)


            tool_name = result.get("tool")
            tool_name = result.get("tool")

            tool_obj = next((t for t in tools if t.name == tool_name), None)

            if tool_obj and tool_obj.input_schema:
                # Always pass raw user input to first parameter
                first_param = list(tool_obj.input_schema.keys())[0]
                args = {first_param: user_input}
            else:
                args = result.get("args", {})


            logger.info(f"[PLANNER] Tool chosen: {tool_name}")
            logger.info(f"[PLANNER] Confidence: {result.get('confidence')}")

            step = PlanStep(
                action=ToolCall(
                    id=str(uuid.uuid4()),
                    tool_name=tool_name,
                    arguments=args,
                ),
                reasoning=result.get("reasoning", "LLM decision"),
                confidence=float(result.get("confidence", 0.7)),
            )

            return Plan(steps=[step], goal="LLM-driven planning")

        except Exception as e:
            logger.error(f"[PLANNER ERROR] JSON parsing failure: {e}")
            logger.error(f"[PLANNER] Raw output: {text}")

            return Plan(steps=[], goal="Planner failed")
