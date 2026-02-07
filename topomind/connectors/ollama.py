import requests
from typing import Dict, Any
from .base import ExecutionConnector


class OllamaConnector(ExecutionConnector):

    def __init__(self, model: str = "mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def execute(self, tool_name: str, args: Any, **kwargs) -> Dict[str, Any]:
        """
        args may be:
        - {"question": "..."}  (correct)
        - "some text"          (LLM mistake)
        - None
        """

        # Normalize input robustly
        if isinstance(args, dict):
            question = args.get("question")
        elif isinstance(args, str):
            question = args
        else:
            question = None

        # Final fallback
        if not question:
            question = f"Explain: {tool_name}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=180,
                proxies={"http": None, "https": None},
            )
            response.raise_for_status()

            data = response.json()
            answer = data.get("message", {}).get("content", "").strip()

            return {"answer": answer or "No response generated."}

        except Exception as e:
            return {"answer": f"LLM reasoning failed: {str(e)}"}
