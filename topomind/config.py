class AgentConfig:
    """
    Central configuration object for agent behavior.
    Controls planner type and LLM backend.
    """

    def __init__(
        self,
        planner_type: str = "rule",   # "rule" or "llm"
        model: str = None,
        llm_backend: str = "ollama",  # "ollama", "groq", or "cohere"
    ):
        self.planner_type = planner_type
        self.model = model
        self.llm_backend = llm_backend

        self._validate()

    def _validate(self):
        if self.planner_type not in {"rule", "llm"}:
            raise ValueError(f"Unsupported planner_type: {self.planner_type}")

        if self.llm_backend not in {"ollama", "groq", "cohere"}:
            raise ValueError(f"Unsupported llm_backend: {self.llm_backend}")

        if self.planner_type == "llm" and not self.model:
            raise ValueError("LLM planner requires a model name")
