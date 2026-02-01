import json
import requests
from ...planner.interface import ReasoningEngine


class OllamaPlanner(ReasoningEngine):
    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

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

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }

        print("[DEBUG] Sending request to Ollama...")

        response = requests.post(
            self.url,
            json=payload,
            proxies={"http": None, "https": None},  
        )

        data = response.json()
        text = data.get("message", {}).get("content", "").strip()

        print("[DEBUG] Raw LLM output:", text)

        try:
            return json.loads(text)
        except:
            return {"tool": "echo", "args": {"text": "Planner failed"}}
