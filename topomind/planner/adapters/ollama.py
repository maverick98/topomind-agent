import json
import uuid
import requests
import logging
import time
from typing import List

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

    def __init__(self, model: str = "mistral"):
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

        logger.debug("----- PLANNER PROMPT -----")
        logger.debug(prompt)

        # -------------------------------------------------------
        # STRICT MODE TEMPERATURE CONTROL
        # -------------------------------------------------------

        strict_mode_enabled = any(getattr(t, "strict", False) for t in tools)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        if strict_mode_enabled:
            logger.info("[PLANNER] Strict tool detected. Forcing temperature=0.")
            payload["temperature"] = 0

        # -------------------------------------------------------
        # DEBUG: Ollama Call Visibility
        # -------------------------------------------------------

        logger.info("[PLANNER] Sending request to Ollama...")
        start_time = time.time()

        try:
            response = requests.post(
                self.url,
                json=payload,
                proxies={"http": None, "https": None},
            )
        except Exception as e:
            logger.error(f"[PLANNER] Exception while contacting Ollama: {e}")
            raise

        elapsed = time.time() - start_time
        logger.info(f"[PLANNER] Ollama responded in {elapsed:.2f}s")

        # -------------------------------------------------------

        try:
            response_json = response.json()
        except Exception as e:
            logger.error(f"[PLANNER] Failed to decode Ollama JSON response: {e}")
            logger.error(f"[PLANNER] Raw response text: {response.text}")
            raise

        text = response_json.get("message", {}).get("content", "").strip()

        logger.debug(f"[PLANNER] Raw LLM output:\n{text}")

        try:
            result = json.loads(text)

            tool_name = result.get("tool", "echo")
            args = result.get("args", {})

            logger.info(f"[PLANNER] Tool chosen: {tool_name}")
            logger.info(f"[PLANNER] Confidence: {result.get('confidence')}")
            logger.info(f"[PLANNER] Reasoning: {result.get('reasoning')}")

            if not args:
                tool_obj = next((t for t in tools if t.name == tool_name), None)
                if tool_obj and tool_obj.input_schema:
                    first_param = list(tool_obj.input_schema.keys())[0]
                    args = {first_param: user_input}

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
            logger.error(f"[PLANNER] Failed planner output: {text}")

            step = PlanStep(
                action=ToolCall(
                    id=str(uuid.uuid4()),
                    tool_name="echo",
                    arguments={"text": "Planner failed"},
                ),
                reasoning="Fallback due to LLM parse failure.",
                confidence=0.2,
            )

            return Plan(steps=[step], goal="Fallback")
