from collections import Counter


class PersistenceAnalyzer:
    def __init__(self, memory_graph):
        self.memory = memory_graph

    def persistent_entities(self, threshold=2):
        values = [n.value for n in self.memory.get_nodes_by_type("entity")]
        counts = Counter(values)
        return [entity for entity, c in counts.items() if c >= threshold]
