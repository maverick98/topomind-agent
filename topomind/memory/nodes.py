from dataclasses import dataclass
from typing import Any


@dataclass
class Node:
    id: str
    type: str  # entity, goal, constraint, result, assumption
    value: Any
    turn_created: int
