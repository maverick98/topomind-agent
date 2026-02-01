from dataclasses import dataclass, field
from typing import Dict, Any, List


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

    Design Principles
    -----------------
    - Declarative: contains no execution logic
    - Deterministic: immutable once created
    - Versioned: enables schema evolution without breaking memory
    - Policy-aware: carries runtime constraints (timeouts, retries)
    - Observable: tagged for monitoring and reliability analysis

    Attributes
    ----------
    name : str
        Unique identifier of the tool. Used for planning and dispatch.

    description : str
        Human- and model-readable explanation of tool purpose.

    connector_name : str
        Name of the Connector responsible for execution.

    input_schema : Dict[str, Any]
        Structured definition of expected arguments.

    output_schema : Dict[str, Any]
        Structured definition of returned data.

    version : str
        Semantic version of the tool contract.

    timeout_seconds : int
        Maximum allowed execution time.

    retryable : bool
        Indicates whether transient failures may be retried.

    max_retries : int
        Maximum retry attempts if retryable.

    side_effect : bool
        Whether tool mutates external state.

    tags : List[str]
        Classification and observability tags.
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

    tags: List[str] = field(default_factory=list)

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
