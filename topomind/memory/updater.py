from ..models import Observation
from .graph import MemoryGraph
from .dedup import MemoryDeduplicator
from .persistence_score import PersistenceScorer
from .decay import MemoryDecay
from .forgetting import MemoryForgetting


class MemoryUpdater:
    """
    Central orchestrator for the memory lifecycle.

    This component is the "hippocampus" of the agent — it controls how
    new information is encoded, reinforced, decayed, and forgotten.

    Responsibilities
    ----------------
    • Convert observations into memory nodes
    • Deduplicate knowledge to avoid graph explosion
    • Track reinforcement frequency (persistence scoring)
    • Apply time-based decay to memory strength
    • Periodically prune low-importance memories
    • Serve as the single owner of persistence scoring state

    Architectural Note
    -------------------
    MemoryGraph stores structure.
    MemoryUpdater manages *memory dynamics*.
    """

    # Forgetting cycle frequency (turn-based)
    PRUNE_INTERVAL = 5
    PRUNE_THRESHOLD = -5

    def __init__(self, graph: MemoryGraph):
        self._graph = graph

        # Structural layer
        self._dedup = MemoryDeduplicator(graph)

        # Cognitive dynamics
        self._scorer = PersistenceScorer()
        self._decay = MemoryDecay(graph, self._scorer)
        self._forget = MemoryForgetting(graph, self._decay)

    # ------------------------------------------------------------------
    # Persistence Access (Ownership Boundary)
    # ------------------------------------------------------------------

    @property
    def scorer(self) -> PersistenceScorer:
        """
        Expose scorer for persistence loading/saving without
        transferring ownership.
        """
        return self._scorer

    # ------------------------------------------------------------------
    # Memory Ingestion
    # ------------------------------------------------------------------

    def update_from_observation(self, obs: Observation) -> None:
        """
        Integrate a structured observation into memory.

        Flow:
        Observation → Dedup → Node creation/reuse → Reinforcement → Forgetting cycle
        """

        # Deduplicate knowledge
        existing = self._dedup.find_existing(obs.type, obs.payload)
        node_id = existing or self._graph.add_node(obs.type, obs.payload)

        # Reinforce importance
        self._scorer.register_occurrence(node_id)

        # Periodic forgetting cycle
        if self._graph.current_turn % self.PRUNE_INTERVAL == 0:
            self._forget.prune(threshold=self.PRUNE_THRESHOLD)
