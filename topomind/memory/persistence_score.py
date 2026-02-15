from collections import defaultdict
from typing import Dict


class PersistenceScorer:
    """
    Tracks node reinforcement across turns.

    Responsibilities
    ----------------
    • Count reinforcement frequency
    • Track last-seen turn for recency modeling
    • Provide safe export/import
    • Support cleanup of removed nodes

    NOTE:
    This class does NOT implement decay logic.
    Decay policies are handled by MemoryDecay.
    """

    def __init__(self):
        self._scores: Dict[str, int] = defaultdict(int)
        self._last_seen: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------

    def register_occurrence(self, node_id: str, current_turn: int = None) -> None:
        """
        Register reinforcement for a node.

        Parameters
        ----------
        node_id : str
            Node being reinforced.

        current_turn : int (optional)
            If provided, updates last-seen timestamp.
        """
        self._scores[node_id] += 1

        if current_turn is not None:
            self._last_seen[node_id] = current_turn

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def score(self, node_id: str) -> int:
        return self._scores.get(node_id, 0)

    def last_seen(self, node_id: str) -> int:
        return self._last_seen.get(node_id, -1)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove(self, node_ids):
        """
        Remove persistence entries for deleted nodes.
        """
        for nid in node_ids:
            self._scores.pop(nid, None)
            self._last_seen.pop(nid, None)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export(self):
        """
        Export internal state for persistence layer.
        """
        return {
            "scores": dict(self._scores),
            "last_seen": dict(self._last_seen),
        }

    def load(self, data):
        """
        Load persisted state safely.
        """

        if not isinstance(data, dict):
            raise ValueError("Persistence data must be dictionary.")

        scores = data.get("scores", {})
        last_seen = data.get("last_seen", {})

        if not isinstance(scores, dict) or not isinstance(last_seen, dict):
            raise ValueError("Invalid persistence structure.")

        # Defensive validation
        for k, v in scores.items():
            if not isinstance(k, str) or not isinstance(v, int) or v < 0:
                continue
            self._scores[k] = v

        for k, v in last_seen.items():
            if not isinstance(k, str) or not isinstance(v, int) or v < 0:
                continue
            self._last_seen[k] = v
