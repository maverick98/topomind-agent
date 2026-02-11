import requests
from typing import Dict, Any
from .base import ExecutionConnector


DEFAULT_EXECUTION_MODEL = "mistral:latest"


class OllamaConnector(ExecutionConnector):

    def __init__(self, default_model: str = DEFAULT_EXECUTION_MODEL):
        self.default_model = default_model
        self.url = "http://localhost:11434/api/chat"

    def execute(self, tool, args: Any, timeout: int = 180, **kwargs) -> Dict[str, Any]:
        """
        tool: Full Tool object
        args: Validated arguments
        """

        # -------------------------------
        # Model Routing
        # -------------------------------
        model_to_use = tool.execution_model or self.default_model

        print(f"[OllamaConnector] Using model: {model_to_use}")

        # -------------------------------
        # Build Execution Prompt
        # -------------------------------

        # tool.prompt = DSL contract
        execution_contract = tool.prompt.strip() if tool.prompt else ""

        if isinstance(args, dict):
            user_input = str(args)
        elif isinstance(args, str):
            user_input = args
        else:
            user_input = ""

        full_prompt = f"{execution_contract}\n\nUser Input:\n{user_input}"

        payload = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": False,
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=timeout,
                proxies={"http": None, "https": None},
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("message", {}).get("content", "").strip()

            if not content:
                raise RuntimeError("Empty LLM response")

            return content  # raw output for OutputValidator

        except Exception as e:
            raise RuntimeError(f"Ollama execution failed: {str(e)}")
