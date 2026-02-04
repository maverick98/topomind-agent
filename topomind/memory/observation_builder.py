from ..models import Observation
from .semantic_extractor import SemanticExtractor


class ObservationBuilder:

    def __init__(self):
        self.extractor = SemanticExtractor()

    def from_reason_result(self, answer_text: str):
        semantics = self.extractor.extract(answer_text)
        observations = []

        # Concepts
        for c in semantics.get("concepts", []):
            observations.append(
                Observation(
                    source="inference",
                    type="concept",
                    payload=c,
                    metadata={}
                )
            )

        # Facts
        for f in semantics.get("facts", []):
            observations.append(
                Observation(
                    source="inference",
                    type="fact",
                    payload=f,
                    metadata={}
                )
            )

        # Relations
        for r in semantics.get("relations", []):
            observations.append(
                Observation(
                    source="inference",
                    type="relation",
                    payload=(r["source"], r["relation"], r["target"]),
                    metadata={}
                )
            )

        return observations
