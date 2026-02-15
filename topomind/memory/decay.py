class MemoryDecay:
    """
    Computes importance score with reinforcement + recency penalty.

    Importance Model
    ----------------
    • Reinforcement increases importance
    • Recency reduces penalty
    • Older and unused nodes gradually fade
    """

    def __init__(self, graph, scorer):
        self._graph = graph
        self._scorer = scorer

    def compute_importance(self, node_id: str) -> float:

        node = self._graph.get_node(node_id)

        current_turn = self._graph.current_turn

        # ----------------------------------------------------------
        # Recency (use last_seen, not creation turn)
        # ----------------------------------------------------------

        last_seen = self._scorer.last_seen(node_id)
        age = current_turn - last_seen if last_seen >= 0 else current_turn - node.turn_created

        # ----------------------------------------------------------
        # Reinforcement (diminishing returns)
        # ----------------------------------------------------------

        persistence = self._scorer.score(node_id)

        reinforcement_score = persistence ** 0.5  # sqrt dampening

        # ----------------------------------------------------------
        # Linear decay penalty
        # ----------------------------------------------------------

        decay_penalty = age * 0.15

        return reinforcement_score - decay_penalty
