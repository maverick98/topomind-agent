import os
import requests
from typing import Optional

from .base import ExecutionConnector


class GroqConnector(ExecutionConnector):
    """
    Groq LLM connector.

    Responsible ONLY for sending prompts to Groq and
    returning raw text output.

    Tool orchestration, schema validation, and prompt
    construction are handled by ToolExecutor.
    """

    def __init__(
        self,
        model: str | None = None,
        default_model: str = "llama-3.1-8b-instant",
    ):
        # Support legacy `model=` usage from app.py
        self.default_model = model or default_model

        self.url = "https://api.groq.com/openai/v1/chat/completions"

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")


    # ------------------------------------------------------------
    # LLM execution
    # ------------------------------------------------------------
    def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: int = 30,
    ) -> str:
        """
        Executes a prompt against Groq.

        Parameters:
            prompt: Fully rendered prompt string
            model: Optional override model name
            timeout: HTTP timeout in seconds

        Returns:
            Raw string response from LLM
        """

        model_to_use = model or self.default_model

        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
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

        data = response.json()

        return data["choices"][0]["message"]["content"]
