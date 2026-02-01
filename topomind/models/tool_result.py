from dataclasses import dataclass
from typing import Any, Optional, Literal


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
