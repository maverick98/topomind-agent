import os
import requests
from .llm_client import LLMClient


class GroqClient(LLMClient):

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set"
            )

    def chat(self, prompt: str, strict: bool) -> str:

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0 if strict else 0.7,
        }

        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
