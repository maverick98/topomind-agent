from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsEntityNode(Protocol):
    """Protocol for memory nodes that may represent entities."""
    value: Any


@runtime_checkable
class SupportsMemoryGraph(Protocol):
    """Protocol describing the required memory graph interface."""
    def get_nodes_by_type(self, type_name: str) -> Iterable[SupportsEntityNode]: ...


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

    def persistent_entities(self, threshold: int = 2) -> List[Any]:
        """
        Return entity values appearing at least `threshold` times.

        Parameters
        ----------
        threshold : int
            Minimum occurrences required to consider an entity stable.

        Returns
        -------
        List[Any]
            Stable entity values safe for planner biasing.
        """
        if threshold < 1:
            raise ValueError("threshold must be >= 1")

        nodes = self._safe_entity_nodes()
        values = self._extract_hashable_values(nodes)

        if not values:
            return []

        counts = Counter(values)
        return [value for value, count in counts.items() if count >= threshold]

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

    def _extract_hashable_values(
        self, nodes: Iterable[SupportsEntityNode]
    ) -> List[Any]:
        """Extract valid, hashable entity values."""
        values: List[Any] = []

        for node in nodes:
            value = getattr(node, "value", None)
            if value is None:
                continue

            try:
                hash(value)
            except TypeError:
                continue

            values.append(value)

        return values
