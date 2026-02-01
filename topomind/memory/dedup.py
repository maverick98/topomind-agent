from typing import Optional
from .graph import MemoryGraph


class MemoryDeduplicator:
    """
    Prevents redundant nodes from being created.

    NOTE: O(n) scan. Can be replaced with indexing later.
    """

    def __init__(self, graph: MemoryGraph):
        self._graph = graph

    def find_existing(self, type_: str, value) -> Optional[str]:
        for node in self._graph.nodes():
            if node.type == type_ and node.value == value:
                return node.id
        return None
