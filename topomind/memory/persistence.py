import json
from .graph import MemoryGraph
from .nodes import Node
from .edges import Edge


class MemoryPersistence:
    """
    Handles serialization and deserialization of memory graph
    and persistence scoring state.
    """

    @staticmethod
    def save(graph: MemoryGraph, scorer, path: str) -> None:
        data = {
            "turn": graph.current_turn,
            "nodes": [node.__dict__ for node in graph.nodes()],
            "edges": [edge.__dict__ for edge in graph.edges()],
            "scores": scorer.export(),
        }

        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(graph: MemoryGraph, scorer, path: str) -> None:
        with open(path) as f:
            data = json.load(f)

        nodes = {n["id"]: Node(**n) for n in data["nodes"]}
        edges = [Edge(**e) for e in data["edges"]]

        graph.load_state(data["turn"], nodes, edges)
        scorer.load(data.get("scores", {}))
