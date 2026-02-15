from dataclasses import dataclass
from typing import Any, Optional, Literal, Dict


@dataclass(frozen=True)
class ToolResult:
    """
    Immutable structured record of a tool execution.

    This object represents a single interaction between the agent
    and an external system via a Tool. It is the canonical execution
    observation passed into memory, stability monitoring, and reasoning.

    A ToolResult is version-aware, ensuring that outputs remain tied
    to the schema contract that produced them. This enables schema
    migration, replay, and long-term system stability.

    Attributes
    ----------
    tool_name : str
        Name of the tool that was executed.

    tool_version : str
        Version of the tool schema used during execution.

    status : {"success", "failure", "blocked"}
        Outcome classification:
            success → tool executed correctly
            failure → tool ran but returned error
            blocked → execution prevented (policy/lookup issue)

    output : Any
        Structured tool output (validated). None if execution failed.

    error : Optional[str]
        Error message when status is failure or blocked.

    latency_ms : int
        Execution time in milliseconds (monotonic).

    stability_signal : float
        Confidence score (0.0–1.0) used by stability system.
        Decreases with retries, anomalies, or degraded performance.
    """

    tool_name: str
    tool_version: str
    status: Literal["success", "failure", "blocked"]
    output: Any
    error: Optional[str]
    latency_ms: int
    stability_signal: float

    # ------------------------------------------------------------------
    # Post Init Normalization
    # ------------------------------------------------------------------

    def __post_init__(self):
        # Clamp stability signal safely into range
        object.__setattr__(
            self,
            "stability_signal",
            max(0.0, min(1.0, float(self.stability_signal)))
        )

    # ------------------------------------------------------------------
    # Convenience Properties
    # ------------------------------------------------------------------

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def is_failure(self) -> bool:
        return self.status == "failure"

    @property
    def is_blocked(self) -> bool:
        return self.status == "blocked"

    # ------------------------------------------------------------------
    # Safe Serialization Boundary (NEW)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-safe dictionary.

        This method should be used when returning results
        outside the agent boundary (e.g., FastAPI layer).
        """

        return {
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "stability_signal": self.stability_signal,
        }
