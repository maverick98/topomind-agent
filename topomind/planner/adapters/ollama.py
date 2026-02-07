import json
import uuid
import requests
import logging
from typing import List

from ..interface import ReasoningEngine
from ..plan_model import Plan, PlanStep
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

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:

        tool_desc = "\n".join(
            [f"- {t.name}: {t.description}, inputs={t.input_schema}" for t in tools]
        )

        prompt = f"""
You are the planning engine of an AI agent.

Your job is to select the SINGLE most appropriate tool to handle the user request.

Choose the tool whose description best matches the task.

IMPORTANT:
If stable context contains "previous_tool" or "previous_error",
it means the previous tool choice failed.
You MUST choose a DIFFERENT tool.

You DO NOT generate answers. You ONLY choose tools.

Return STRICT JSON:
{{ "tool": "...", "args": {{...}}, "reasoning": "...", "confidence": 0.0-1.0 }}

User request:
"{user_input}"

Stable context:
{signals}

Available tools:
{tool_desc}
"""

        logger.debug("----- PLANNER PROMPT -----")
        logger.debug(prompt)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        response = requests.post(
            self.url,
            json=payload,
            proxies={"http": None, "https": None},
        )

        text = response.json().get("message", {}).get("content", "").strip()

        logger.debug(f"Planner raw LLM output: {text}")

        try:
            result = json.loads(text)
            print(f"result as recvd {result}")
            tool_name = result.get("tool", "echo")
            print(f"tool_name is {tool_name}")
            # CRITICAL: LLM chooses tool, system controls arguments
            if tool_name == "reason":
                args = {"question": user_input}
            else:
                args = result.get("args", {})

            logger.info(f"[PLANNER] Tool chosen: {result.get('tool')}")
            logger.info(f"[PLANNER] Confidence: {result.get('confidence')}")
            logger.info(f"[PLANNER] Reasoning: {result.get('reasoning')}")

            tool_name = result.get("tool", "echo")
            args = result.get("args", {})

            print("\n🧠 ================= PLANNER DECISION =================")
            print(f"User Input   : {user_input}")
            print(f"Signals      : {signals}")
            print(f"Chosen Tool  : {tool_name}")
            print(f"Arguments    : {args}")
            print(f"Confidence   : {result.get('confidence')}")
            print(f"Reasoning    : {result.get('reasoning')}")
            print("======================================================\n")


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
            logger.error(f"[PLANNER ERROR] {e}")
            logger.error(f"Failed planner output: {text}")

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
