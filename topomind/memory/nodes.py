from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Node:
    """
    Immutable unit of knowledge stored in the agent memory graph.

    Nodes represent semantic concepts such as:
    entity, goal, constraint, result, assumption, signal.

    Nodes are immutable, but the graph may prune them through
    controlled forgetting policies.
    """

    id: str
    type: str
    value: Any
    turn_created: int
