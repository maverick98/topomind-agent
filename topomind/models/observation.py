from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Observation:
    """
    A normalized perception unit inside the agent.

    Everything the agent learns (tool outputs, memory reads,
    system signals) becomes an Observation before integration
    into the memory graph or reasoning loop.
    """

    source: str      # "tool", "memory", "system"
    type: str        # "data", "error", "signal"
    payload: Any
    metadata: Dict[str, Any]
