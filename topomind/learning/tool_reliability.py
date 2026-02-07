class ToolReliability:

    def __init__(self):
        self.stats = {}

    def record(self, tool_name: str, success: bool):
        if tool_name not in self.stats:
            self.stats[tool_name] = {"success": 0, "fail": 0}

        if success:
            self.stats[tool_name]["success"] += 1
        else:
            self.stats[tool_name]["fail"] += 1

    def score(self, tool_name: str) -> float:
        s = self.stats.get(tool_name, {"success": 0, "fail": 0})
        total = s["success"] + s["fail"]
        if total == 0:
            return 0.5  # neutral default
        return s["success"] / total

    def all_scores(self):
        return {t: self.score(t) for t in self.stats}
