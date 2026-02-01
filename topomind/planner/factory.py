from .rule_planner import RuleBasedPlanner
from .interface import ReasoningEngine


def create_planner(config) -> ReasoningEngine:
    """
    Factory for constructing the system reasoning engine.

    Planner selection is driven by configuration. Supported types:

    - "rule"    → deterministic rule-based planner
    - "ollama"  → LLM planner using local Ollama models
    - "openai"  → LLM planner using OpenAI API

    Raises
    ------
    ValueError
        If planner type is unknown.
    """

    planner_type = getattr(config, "planner_type", "rule")

    if planner_type == "rule":
        return RuleBasedPlanner()

    if planner_type == "ollama":
        from .adapters.ollama import OllamaPlanner
        return OllamaPlanner(model=config.model)

    if planner_type == "openai":
        from .adapters.openai import OpenAIPlanner
        return OpenAIPlanner(model=config.model)

    raise ValueError(f"Unsupported planner type: {planner_type}")
