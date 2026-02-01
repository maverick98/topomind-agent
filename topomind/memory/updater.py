from .graph import MemoryGraph


class MemoryUpdater:
    def __init__(self, graph: MemoryGraph):
        self.graph = graph

    def update_from_input(self, user_input: str):
        self.graph.new_turn()
        self.graph.add_node("entity", user_input)
