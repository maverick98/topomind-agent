class MemoryForgetting:
    """
    Removes low-importance episodic nodes while preserving
    structural memory integrity.

    Structural node types (goal, constraint, signal) are never
    pruned automatically because they represent agent state
    and policy context.

    Enhancements
    ------------
    • Observability: logs pruned nodes
    • Safety: batch pruning cap
    • Stable structural guarantees
    """

    # Node types that should NEVER be auto-forgotten
    PROTECTED_TYPES = {"goal", "constraint", "signal"}

    # Safety: max nodes removed per pruning cycle
    MAX_PRUNE_BATCH = 25

    def __init__(self, graph, decay):
        self._graph = graph
        self._decay = decay

    def prune(self, threshold: float = 0.0) -> None:

        to_remove = []

        for node in list(self._graph.nodes()):

            # ------------------------------------------------------
            # Protect structural memory
            # ------------------------------------------------------
            if node.type in self.PROTECTED_TYPES:
                continue

            importance = self._decay.compute_importance(node.id)

            if importance < threshold:
                to_remove.append((node.id, importance))

        if not to_remove:
            return

        # ----------------------------------------------------------
        # Sort by lowest importance first (most irrelevant first)
        # ----------------------------------------------------------

        to_remove.sort(key=lambda x: x[1])

        # ----------------------------------------------------------
        # Apply safety cap
        # ----------------------------------------------------------

        pruned_ids = [nid for nid, _ in to_remove[:self.MAX_PRUNE_BATCH]]

        removed_nodes, removed_edges = self._graph.remove_nodes(pruned_ids)

        # ----------------------------------------------------------
        # Observability (optional — safe no-op if no logger)
        # ----------------------------------------------------------

        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"[FORGETTING] Removed {len(removed_nodes)} nodes "
                f"and {len(removed_edges)} edges."
            )
        except Exception:
            pass
