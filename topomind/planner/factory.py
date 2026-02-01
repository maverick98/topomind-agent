from .rule_planner import RuleBasedPlanner


def create_planner(config):
    if config.planner_type == "ollama":
        from .adapters.ollama import OllamaPlanner
        return OllamaPlanner(model=config.model)

    if config.planner_type == "openai":
        from .adapters.openai import OpenAIPlanner
        return OpenAIPlanner(model=config.model)

    return RuleBasedPlanner()
