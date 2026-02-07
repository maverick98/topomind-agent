class SemanticExtractor:
    """
    Lightweight structured extractor.
    Keeps format but avoids LLM call.
    """

    def __init__(self, model="mistral"):
        self.model = model  # reserved for future use

    def extract(self, text: str):
        sentences = [s.strip() for s in text.split('.') if len(s) > 40]

        return {
            "concepts": [],
            "facts": sentences[:3] if sentences else [text],
            "relations": []
        }
