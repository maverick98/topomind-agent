from dataclasses import dataclass, field
from typing import Any, Dict
import time
import uuid


@dataclass(frozen=True)
class Observation:
    """
    A normalized perception unit inside the agent.

    Everything the agent learns (tool outputs, memory reads,
    user input, or system signals) becomes an Observation
    before integration into the memory graph.

    Architectural Role
    ------------------
    Execution → Observation → MemoryUpdater → MemoryGraph

    This abstraction ensures:
    • Consistent memory ingestion
    • Replayability
    • Traceability across cognitive steps
    """

    source: str
    """
    Origin of the observation.

    Examples:
    "user", "tool", "memory", "system"
    """

    type: str
    """
    Semantic category of the observation.

    Examples:
    "entity", "result", "error", "signal"
    """

    payload: Any
    """The actual content observed."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional structured metadata."""

    # --- System Metadata ---
    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when observation was created."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier linking this observation to system events."""

    # ------------------------------------------------------------------
    # Debug Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Observation(type={self.type}, source={self.source}, "
            f"id={self.trace_id[:8]})"
        )
