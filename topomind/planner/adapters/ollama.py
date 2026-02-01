import json
import uuid
import requests
from typing import List

from ..interface import ReasoningEngine
from ..plan_model import Plan, PlanStep
from ...models.tool_call import ToolCall
from ...tools.schema import Tool


class OllamaPlanner(ReasoningEngine):

    def __init__(self, model: str = "mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def generate_plan(self, user_input: str, signals, tools: List[Tool]) -> Plan:
        tool_desc = "\n".join(
            [f"- {t.name}: {t.description}, inputs={t.input_schema}" for t in tools]
        )

        prompt = f"""
You are a planning engine.

User input: "{user_input}"
Stable context: {signals}

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

        response = requests.post(self.url, json=payload, proxies={"http": None, "https": None})
        text = response.json().get("message", {}).get("content", "").strip()

        try:
            result = json.loads(text)

            step = PlanStep(
                action=ToolCall(
                    id=str(uuid.uuid4()),
                    tool_name=result["tool"],
                    arguments=result["args"],
                ),
                reasoning=result.get("reasoning", "LLM decision"),
                confidence=float(result.get("confidence", 0.7)),
            )

            return Plan(steps=[step], goal="LLM-driven planning")

        except Exception:
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
