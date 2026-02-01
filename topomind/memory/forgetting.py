class MemoryForgetting:
    """
    Removes low-importance episodic nodes while preserving
    structural memory integrity.

    Structural node types (goal, constraint, signal) are never
    pruned automatically because they represent agent state
    and policy context.
    """

    # Node types that should NEVER be auto-forgotten
    PROTECTED_TYPES = {"goal", "constraint", "signal"}

    def __init__(self, graph, decay):
        self._graph = graph
        self._decay = decay

    def prune(self, threshold: float = 0.0) -> None:
        to_remove = []

        for node in list(self._graph.nodes()):

            #  Protect structural memory
            if node.type in self.PROTECTED_TYPES:
                continue

            importance = self._decay.compute_importance(node.id)

            if importance < threshold:
                to_remove.append(node.id)

        if to_remove:
            self._graph.remove_nodes(to_remove)
