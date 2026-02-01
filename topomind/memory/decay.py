class MemoryDecay:
    """
    Computes importance score with reinforcement + age penalty.
    """

    def __init__(self, graph, scorer):
        self._graph = graph
        self._scorer = scorer

    def compute_importance(self, node_id: str) -> float:
        node = self._graph.get_node(node_id)
        age = self._graph.current_turn - node.turn_created
        persistence = self._scorer.score(node_id)

        return (persistence * 2) - (age * 0.1)
