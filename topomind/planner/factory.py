from .rule_planner import RuleBasedPlanner
from .interface import ReasoningEngine
from topomind.config import AgentConfig
from .adapters.llm_planner import LLMPlanner

def create_planner(config: AgentConfig) -> ReasoningEngine:
    """
    Factory for constructing the system reasoning engine.

    Planner selection is driven by configuration.

    Supported planner types:
    - "rule" → deterministic rule-based planner
    - "llm"  → LLM-based planner (backend selectable)

    LLM backends:
    - "ollama"
    - "groq"
    """

    planner_type = config.planner_type

    # ---------------------------------------------------------
    # RULE-BASED PLANNER
    # ---------------------------------------------------------
    if planner_type == "rule":
        return RuleBasedPlanner()

    # ---------------------------------------------------------
    # LLM-BASED PLANNER
    # ---------------------------------------------------------
    if planner_type == "llm":

        if not config.model:
            raise ValueError(
                "LLM planner requires a model name in AgentConfig."
            )

        if not config.llm_backend:
            raise ValueError(
                "LLM planner requires llm_backend in AgentConfig."
            )

        



        # Lazy imports prevent unnecessary dependency loading
        if config.llm_backend == "ollama":
            from topomind.agent.llm import OllamaClient
            client = OllamaClient(model=config.model)

        elif config.llm_backend == "groq":
            from topomind.agent.llm import GroqClient
            client = GroqClient(model=config.model)

        else:
            raise ValueError(
                f"Unsupported llm_backend: {config.llm_backend}"
            )

        return LLMPlanner(client)

    # ---------------------------------------------------------
    # UNKNOWN PLANNER TYPE
    # ---------------------------------------------------------
    raise ValueError(
        f"Unsupported planner_type: {planner_type}"
    )
