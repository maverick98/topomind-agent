from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsEntityNode(Protocol):
    """Protocol for memory nodes that may represent entities."""
    value: Any
    turn_created: int


@runtime_checkable
class SupportsMemoryGraph(Protocol):
    """Protocol describing the required memory graph interface."""
    def get_nodes_by_type(self, type_name: str) -> Iterable[SupportsEntityNode]: ...
    @property
    def current_turn(self) -> int: ...


class PersistenceAnalyzer:
    """
    Detects entities that persist across memory observations.

    Purpose
    -------
    This component helps the agent distinguish:
    - transient mentions (short-term)
    - stable concepts (long-term relevance)

    Persistence is currently frequency-based but can evolve to
    recency- or weight-based scoring without changing this API.
    """

    ENTITY_TYPE = "entity"

    def __init__(self, memory_graph: SupportsMemoryGraph) -> None:
        self._memory = memory_graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def persistent_entities(
        self,
        threshold: int = 2,
        minimum_turn_age: int = 1,
    ) -> List[Any]:
        """
        Return entity values appearing at least `threshold` times
        and surviving at least `minimum_turn_age` turns.

        Parameters
        ----------
        threshold : int
            Minimum occurrences required to consider an entity stable.

        minimum_turn_age : int
            Minimum number of turns the entity must survive.

        Returns
        -------
        List[Any]
            Stable entity values safe for planner biasing.
        """

        if threshold < 1:
            raise ValueError("threshold must be >= 1")

        nodes = self._safe_entity_nodes()

        values = []
        current_turn = getattr(self._memory, "current_turn", 0)

        for node in nodes:
            value = getattr(node, "value", None)
            turn_created = getattr(node, "turn_created", 0)

            if value is None:
                continue

            try:
                hash(value)
            except TypeError:
                continue

            age = current_turn - turn_created
            if age < minimum_turn_age:
                continue

            values.append(value)

        if not values:
            return []

        counts = Counter(values)

        # Deterministic ordering: most frequent first
        stable = [
            value
            for value, count in sorted(
                counts.items(),
                key=lambda x: (-x[1], x[0])
            )
            if count >= threshold
        ]

        return stable

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_entity_nodes(self) -> Iterable[SupportsEntityNode]:
        """Safely retrieve entity nodes without propagating memory faults."""
        try:
            nodes = self._memory.get_nodes_by_type(self.ENTITY_TYPE)
            return nodes if nodes is not None else []
        except Exception:
            return []
