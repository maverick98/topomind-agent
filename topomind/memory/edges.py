from dataclasses import dataclass


@dataclass
class Edge:
    source: str
    target: str
    relation: str  # supports, refines, derived_from, contradicts
