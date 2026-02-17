from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple
import json
import hashlib


@dataclass(frozen=True)
class Tool:

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
    # Artifact Flow Metadata
    # ------------------------------------------------------------------

    produces: Tuple[str, ...] = field(default_factory=tuple)
    consumes: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Model-Facing Execution Contract
    # ------------------------------------------------------------------

    prompt: str = ""
    strict: bool = False

    # ------------------------------------------------------------------
    # Model Routing
    # ------------------------------------------------------------------

    execution_model: str = ""

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

    tags: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Validation Layer
    # ------------------------------------------------------------------

    def __post_init__(self):

        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tool name must be a non-empty string.")

        if not self.connector_name or not isinstance(self.connector_name, str):
            raise ValueError("Connector name must be a non-empty string.")

        if not isinstance(self.input_schema, dict):
            raise TypeError("input_schema must be a dictionary.")

        if not isinstance(self.output_schema, dict):
            raise TypeError("output_schema must be a dictionary.")

        if not isinstance(self.produces, tuple):
            raise TypeError("produces must be a tuple.")

        if not isinstance(self.consumes, tuple):
            raise TypeError("consumes must be a tuple.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if not isinstance(self.version, str):
            raise TypeError("version must be a string.")

        if self.execution_model and not isinstance(self.execution_model, str):
            raise TypeError("execution_model must be a string.")

        if not self.retryable and self.max_retries > 0:
            raise ValueError("max_retries must be 0 when retryable is False.")

    # ------------------------------------------------------------------
    # Derived Properties
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def is_strict(self) -> bool:
        return self.strict

    # ------------------------------------------------------------------
    # Serialization / Observability
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns canonical dictionary representation of the Tool contract.
        Safe for logging, JSON, hashing, replay.
        """

        data = asdict(self)

        # Convert tuples to lists for JSON safety
        data["produces"] = list(self.produces)
        data["consumes"] = list(self.consumes)
        data["tags"] = list(self.tags)

        return data

    @property
    def contract_hash(self) -> str:
        """
        Stable hash of entire tool contract.
        Detects schema drift across deployments.
        """

        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_debug_string(self) -> str:
        """
        Multi-line full contract dump for debugging.
        """

        prompt_block = self.prompt.strip() if self.prompt else "[EMPTY]"

        return (
            f"\n"
            f"================ TOOL CONTRACT =================\n"
            f"Name              : {self.name}\n"
            f"Version           : {self.version}\n"
            f"Key               : {self.key}\n"
            f"Contract Hash     : {self.contract_hash}\n"
            f"\n"
            f"Connector         : {self.connector_name}\n"
            f"Strict            : {self.strict}\n"
            f"Execution Model   : {self.execution_model or 'SYSTEM DEFAULT'}\n"
            f"\n"
            f"Timeout (sec)     : {self.timeout_seconds}\n"
            f"Retryable         : {self.retryable}\n"
            f"Max Retries       : {self.max_retries}\n"
            f"Side Effect       : {self.side_effect}\n"
            f"\n"
            f"Produces          : {self.produces}\n"
            f"Consumes          : {self.consumes}\n"
            f"Tags              : {self.tags}\n"
            f"\n"
            f"Input Schema      : {self.input_schema}\n"
            f"Output Schema     : {self.output_schema}\n"
            f"\n"
            f"Prompt Length     : {len(self.prompt)}\n"
            f"Prompt:\n"
            f"{prompt_block}\n"
            f"=================================================\n"
        )
