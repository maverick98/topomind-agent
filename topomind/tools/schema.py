from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class Tool:
    """
    Versioned declarative contract describing an agent capability.

    A Tool defines WHAT action can be performed, while execution details
    are delegated to a Connector. This object serves as the canonical
    contract between all runtime layers:

        Planner → ArgumentValidator → ToolExecutor → Connector → Memory

    A Tool is immutable and versioned. Any change to its input or output
    schema constitutes a contract change and must increment `version`.
    Historical versions are preserved for schema migration and replay.

    Enhancements (Backward Compatible)
    -----------------------------------
    - prompt: Model-facing execution contract (optional)
    - strict: Indicates whether planner must strictly obey tool prompt
    """

    # ------------------------------------------------------------------
    # Core Identity
    # ------------------------------------------------------------------

    name: str
    description: str
    connector_name: str

    # ------------------------------------------------------------------
    # Schemas (Contract Layer)
    # ------------------------------------------------------------------

    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

    # ------------------------------------------------------------------
    # Model-Facing Execution Contract (NEW — Optional)
    # ------------------------------------------------------------------

    prompt: str = ""
    strict: bool = False

    # ------------------------------------------------------------------
    # Model Routing (NEW — Optional)
    # ------------------------------------------------------------------

    execution_model: str = ""  # empty = use system default

    # ------------------------------------------------------------------
    # Schema Evolution
    # ------------------------------------------------------------------

    version: str = "1.0.0"

    # ------------------------------------------------------------------
    # Runtime Policy Metadata
    # ------------------------------------------------------------------

    timeout_seconds: int = 10
    retryable: bool = True
    max_retries: int = 2
    side_effect: bool = False

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    # NOTE: Tuple used instead of List to preserve immutability
    tags: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Validation Layer (Safe, Non-Breaking)
    # ------------------------------------------------------------------

    def __post_init__(self):
        """
        Lightweight invariant checks.

        Does NOT modify state (frozen dataclass).
        Raises early if contract is malformed.
        """

        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tool name must be a non-empty string.")

        if not self.connector_name or not isinstance(self.connector_name, str):
            raise ValueError("Connector name must be a non-empty string.")

        if not isinstance(self.input_schema, dict):
            raise TypeError("input_schema must be a dictionary.")

        if not isinstance(self.output_schema, dict):
            raise TypeError("output_schema must be a dictionary.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if not isinstance(self.version, str):
            raise TypeError("version must be a string.")
        
        if self.execution_model and not isinstance(self.execution_model, str):
            raise TypeError("execution_model must be a string.")

        # ------------------------------------------------------------------
        # Consistency Enforcement (Safe, Non-Breaking)
        # ------------------------------------------------------------------

        if not self.retryable and self.max_retries > 0:
            raise ValueError(
                "max_retries must be 0 when retryable is False."
            )

    # ------------------------------------------------------------------
    # Derived Properties
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        """
        Unique identifier combining tool name and version.

        Used by SchemaRegistry to track historical contract versions.
        """
        return f"{self.name}:{self.version}"

    @property
    def is_strict(self) -> bool:
        """
        Indicates whether this tool enforces strict planner adherence.
        """
        return self.strict
