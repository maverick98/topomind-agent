import os
import json
import requests
from typing import Dict, Any

from .base import ExecutionConnector


class GroqConnector(ExecutionConnector):
    """
    Groq LLM execution connector.

    Used when:
    - tool.connector_name == "llm"
    - tool.execution_model is set
    """

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

    # ------------------------------------------------------------
    # Required by ExecutionConnector interface
    # ------------------------------------------------------------
    def execute(
        self,
        tool,
        args: Dict[str, Any],
        timeout: int,
    ) -> Any:

        model_to_use = tool.execution_model or self.model

        # Build final prompt
        base_prompt = tool.prompt or ""
        input_json = json.dumps(args, indent=2)

        final_prompt = f"""
{base_prompt}

Input:
{input_json}

Return STRICT JSON matching this schema:
{json.dumps(tool.output_schema, indent=2)}

No explanation.
"""

        payload = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": 0 if tool.strict else 0.7,
        }

        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        raw = response.json()["choices"][0]["message"]["content"]

        # Try extracting JSON safely
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end + 1])
        except Exception:
            pass

        # Fallback behavior
        if len(tool.output_schema) == 1:
            key = next(iter(tool.output_schema))
            return {key: raw}

        return {"result": raw}
