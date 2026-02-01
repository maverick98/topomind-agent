from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    """
    Directed relationship between two nodes in the memory graph.
    """

    source: str
    target: str
    relation: str
