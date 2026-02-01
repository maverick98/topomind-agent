from collections import defaultdict
from typing import Dict


class PersistenceScorer:
    """
    Tracks node reinforcement across turns.
    """

    def __init__(self):
        self._scores: Dict[str, int] = defaultdict(int)

    def register_occurrence(self, node_id: str) -> None:
        self._scores[node_id] += 1

    def score(self, node_id: str) -> int:
        return self._scores.get(node_id, 0)

    def export(self):
        return dict(self._scores)

    def load(self, data):
        self._scores.update(data)
