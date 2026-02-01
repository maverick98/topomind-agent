import json
from openai import OpenAI
from ...planner.interface import ReasoningEngine


class OpenAIPlanner(ReasoningEngine):
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def generate_plan(self, user_input, signals, tools):
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
{{ "tool": "...", "args": {{...}} }}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.choices[0].message.content.strip()

        try:
            return json.loads(text)
        except:
            return {"tool": "echo", "args": {"text": "Planner failed"}}
