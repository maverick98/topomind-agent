import os
import requests
import logging
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


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

        temperature = 0 if strict else 0.7

        # --------------------------------------------------
        # DEBUG: Outgoing Request
        # --------------------------------------------------
        logger.info("========== GROQ CLIENT ==========")
        logger.info(f"Model: {self.model}")
        logger.info(f"Strict mode: {strict}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Prompt length: {len(prompt)}")
        logger.debug(f"Prompt preview:\n{prompt[:2000]}")
        logger.info("=================================")

        payload = {
            "model": self.model,
            "messages": [
                # System role is usually more stable for long instruction blocks
                {"role": "system", "content": prompt}
            ],
            "temperature": temperature,
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

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # --------------------------------------------------
        # DEBUG: Incoming Response
        # --------------------------------------------------
        logger.info("----- GROQ RESPONSE -----")
        logger.info(content)
        logger.info("-------------------------")

        return content
