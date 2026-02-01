import uuid
from typing import Dict, List
from .nodes import Node
from .edges import Edge


class MemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.turn = 0

    def new_turn(self):
        self.turn += 1

    def add_node(self, type_: str, value):
        node_id = str(uuid.uuid4())
        node = Node(node_id, type_, value, self.turn)
        self.nodes[node_id] = node
        return node_id

    def add_edge(self, source, target, relation):
        self.edges.append(Edge(source, target, relation))

    def get_nodes_by_type(self, type_):
        return [n for n in self.nodes.values() if n.type == type_]
