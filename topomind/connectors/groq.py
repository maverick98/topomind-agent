import os
import requests
import logging
from typing import Optional

from .base import ExecutionConnector

logger = logging.getLogger(__name__)


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

        # ---------------------------------------------
        # DEBUG (Safe + Structured Logging)
        # ---------------------------------------------
        logger.info("========== GROQ CONNECTOR ==========")
        logger.info(f"Model: {model_to_use}")
        logger.info(f"Prompt length: {len(prompt)}")
        logger.debug(f"Prompt preview:\n{prompt[:1500]}")
        logger.info("====================================")

        payload = {
            "model": model_to_use,
            "messages": [
                # IMPORTANT: deterministic compiler prompt behaves
                # more reliably as SYSTEM role
                {"role": "system", "content": prompt}
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

        raw_output = data["choices"][0]["message"]["content"]

        # ---------------------------------------------
        # DEBUG OUTPUT
        # ---------------------------------------------
        logger.info("----- GROQ RAW OUTPUT -----")
        logger.info(raw_output)
        logger.info("---------------------------")

        # Strip <think> blocks if present
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

        return cleaned
