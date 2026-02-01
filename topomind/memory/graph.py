import uuid
from typing import Dict, List, Iterable, Tuple

from .nodes import Node
from .edges import Edge


class MemoryGraph:
    """
    Semantic knowledge graph for agent memory.

    This class is a *data structure only* — it does not implement
    cognitive policies like decay or forgetting. Those belong to
    MemoryUpdater.

    Nodes are immutable. The graph supports controlled pruning
    and safe state restoration.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._turn: int = 0

    # ------------------------------------------------------------------
    # Turn Management
    # ------------------------------------------------------------------

    def new_turn(self) -> None:
        """Advance the logical conversation turn."""
        self._turn += 1

    @property
    def current_turn(self) -> int:
        return self._turn

    # ------------------------------------------------------------------
    # Node Operations
    # ------------------------------------------------------------------

    def add_node(self, type_: str, value) -> str:
        """
        Add a new node to memory.

        Returns
        -------
        str
            ID of the created node.
        """
        node_id = str(uuid.uuid4())
        self._nodes[node_id] = Node(node_id, type_, value, self._turn)
        return node_id

    def get_node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def get_nodes_by_type(self, type_: str) -> List[Node]:
        return [n for n in self._nodes.values() if n.type == type_]

    def nodes(self) -> Iterable[Node]:
        return self._nodes.values()

    # ------------------------------------------------------------------
    # Edge Operations
    # ------------------------------------------------------------------

    def add_edge(self, source: str, target: str, relation: str) -> None:
        """Add a directed relationship between two nodes."""
        self._edges.append(Edge(source, target, relation))

    def edges(self) -> Iterable[Edge]:
        return self._edges

    # ------------------------------------------------------------------
    # Controlled Pruning (Structural Operation Only)
    # ------------------------------------------------------------------

    def remove_nodes(self, node_ids: List[str]) -> Tuple[List[Node], List[Edge]]:
        """
        Remove nodes and all connected edges safely.

        Returns
        -------
        Tuple[List[Node], List[Edge]]
            The removed nodes and edges (for audit or stability tracking).
        """
        removed_nodes: List[Node] = []
        removed_edges: List[Edge] = []

        # Remove nodes
        for nid in node_ids:
            node = self._nodes.pop(nid, None)
            if node:
                removed_nodes.append(node)

        # Remove edges referencing removed nodes
        remaining_edges = []
        for edge in self._edges:
            if edge.source in node_ids or edge.target in node_ids:
                removed_edges.append(edge)
            else:
                remaining_edges.append(edge)

        self._edges = remaining_edges

        return removed_nodes, removed_edges

    # ------------------------------------------------------------------
    # Persistence Support
    # ------------------------------------------------------------------

    def load_state(
        self,
        turn: int,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> None:
        """
        Replace entire graph state during persistence restore.

        This is the ONLY allowed mutation of internal structures
        from outside normal operations.
        """
        self._turn = turn
        self._nodes = nodes
        self._edges = edges
