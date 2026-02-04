import json
import uuid
import requests
from typing import List

from ..interface import ReasoningEngine
from ..plan_model import Plan, PlanStep
from ...models.tool_call import ToolCall
from ...tools.schema import Tool


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
You are the cognitive planning engine of an AI agent.

Your job is to choose the correct tool.

Decision policy:

- Use tool "reason" for:
  * Explanations
  * Knowledge questions
  * Scientific, historical, conceptual topics

- Use other tools ONLY for:
  * Actions
  * Data retrieval
  * Calculations
  * System tasks

- NEVER use "echo" unless user asks to repeat text.

Stable context signals: {signals}

User input:
"{user_input}"

Available tools:
{tool_desc}

Return ONLY JSON:
{{ "tool": "...", "args": {{...}}, "reasoning": "...", "confidence": 0.0-1.0 }}
"""

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

        except Exception:
            # Fallback if planner output cannot be parsed
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
