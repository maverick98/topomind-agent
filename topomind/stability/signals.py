from typing import Dict, Any
from .persistence import PersistenceAnalyzer


class StabilitySignals:
    """
    Aggregates system-level cognitive stability signals.

    These signals are provided to the planner to bias decisions
    toward consistent, long-term context.
    """

    def __init__(self, memory_graph) -> None:
        self._analyzer = PersistenceAnalyzer(memory_graph)

    def extract(self) -> Dict[str, Any]:
        """
        Produce structured stability signals.

        Returns
        -------
        Dict[str, Any]
            Signals safe for planner consumption.
        """
        return {
            "stable_entities": self._analyzer.persistent_entities(),
        }
