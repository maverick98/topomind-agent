from .persistence import PersistenceAnalyzer


class StabilitySignals:
    def __init__(self, memory_graph):
        self.analyzer = PersistenceAnalyzer(memory_graph)

    def extract(self):
        return {
            "stable_entities": self.analyzer.persistent_entities()
        }
