import requests
import json


class SemanticExtractor:
    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def extract(self, text: str):
        prompt = f"""
Extract structured knowledge from the text below.

Return STRICT JSON in this format:

{{
  "concepts": [],
  "facts": [],
  "relations": [
    {{"source": "", "relation": "", "target": ""}}
  ]
}}

Text:
\"\"\"{text}\"\"\"
"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=120,  # semantic pass can take time
                proxies={"http": None, "https": None},
            )
            response.raise_for_status()

            raw = response.json().get("message", {}).get("content", "").strip()

            # Attempt to parse JSON safely
            try:
                data = json.loads(raw)
            except Exception:
                return {"concepts": [], "facts": [text], "relations": []}

            # Ensure required keys exist
            return {
                "concepts": data.get("concepts", []),
                "facts": data.get("facts", []),
                "relations": data.get("relations", []),
            }

        except Exception:
            # Fallback: store raw answer as a fact
            return {"concepts": [], "facts": [text], "relations": []}
