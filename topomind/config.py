class AgentConfig:
    """
    Configuration object for selecting planner backend.
    """

    def __init__(self, planner_type="rule", model=None):
        self.planner_type = planner_type
        self.model = model
