from typing import Dict, Any
from .persistence import PersistenceAnalyzer


class StabilitySignals:
    """
    Aggregates system-level cognitive stability signals.

    These signals are provided to the planner to bias decisions
    toward consistent, long-term context.

    Enhancements
    ------------
    • Includes memory pressure awareness
    • Includes system turn tracking
    • Provides safe, planner-friendly metadata
    """

    def __init__(self, memory_graph) -> None:
        self._graph = memory_graph
        self._analyzer = PersistenceAnalyzer(memory_graph)

    def extract(self) -> Dict[str, Any]:
        """
        Produce structured stability signals.

        Returns
        -------
        Dict[str, Any]
            Signals safe for planner consumption.
        """

        stable_entities = self._analyzer.persistent_entities()

        memory_size = len(list(self._graph.nodes()))
        current_turn = self._graph.current_turn

        # Simple pressure heuristic
        memory_pressure = (
            "high" if memory_size > 200
            else "medium" if memory_size > 100
            else "low"
        )

        return {
            "stable_entities": stable_entities,
            "memory_size": memory_size,
            "memory_pressure": memory_pressure,
            "current_turn": current_turn,
        }
