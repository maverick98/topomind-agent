from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    connector_name: str  # Which connector executes it
