import requests
from typing import Optional

from .llm_client import LLMClient


class OllamaClient(LLMClient):
    """
    Ollama LLM transport client.
    Local model backend.
    """

    def __init__(
        self,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434/api/chat",
        timeout_seconds: int = 30,
    ):
        self.model = model
        self.url = base_url
        self.timeout = timeout_seconds

    # ---------------------------------------------------------
    # Main Chat Interface
    # ---------------------------------------------------------

    def chat(self, prompt: str, strict: bool = False) -> str:

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 256,
            }
        }

        if strict:
            payload["options"]["temperature"] = 0
            payload["format"] = "json"  # Strong JSON enforcement

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.Timeout:
            raise TimeoutError("Ollama request timed out")

        except requests.RequestException as e:
            raise Exception(f"Ollama request failed: {str(e)}")

        try:
            data = response.json()
            return data["message"]["content"]
        except (KeyError, ValueError) as e:
            raise Exception(
                f"Unexpected Ollama response format: {str(e)}"
            )
