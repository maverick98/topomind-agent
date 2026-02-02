from dataclasses import dataclass, field
from typing import Dict, Any
import uuid


@dataclass(frozen=True)
class ToolCall:
    """
    Represents a planner-issued instruction to execute a tool.

    This is the *execution intent packet* passed from the planner
    to the ToolExecutor. It contains no execution logic — only
    declarative intent.

    Architectural Role
    ------------------
    Planner → ToolCall → ToolExecutor

    This separation ensures:
    • Deterministic execution boundaries
    • Safe logging and replay
    • Compatibility with persistence and tracing
    """

    tool_name: str
    """Name of the tool to invoke."""

    arguments: Dict[str, Any]
    """Validated input parameters for the tool."""

    # --- System Metadata ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for tracing this tool call."""

    confidence: float = 1.0
    """
    Planner confidence in this action [0.0–1.0].
    Used for stability analysis and learning signals.
    """

    reason: str = ""
    """Planner explanation for why this tool was selected."""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self):
        # Normalize confidence
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    # ------------------------------------------------------------------
    # Debug Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ToolCall(id={self.id[:8]}, tool='{self.tool_name}', "
            f"confidence={self.confidence:.2f})"
        )
