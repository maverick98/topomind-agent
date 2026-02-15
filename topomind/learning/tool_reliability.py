import math
import time


class ToolReliability:
    """
    Tracks tool reliability using
    - Success ratio
    - Exponential decay (recency weighting)
    - Minimum sample stabilization
    """

    DECAY = 0.98        # per update decay
    MIN_SAMPLES = 5     # before full trust

    def __init__(self):
        self.stats = {}
        self.last_update = {}

    # ------------------------------------------------------------
    # Record Execution Outcome
    # ------------------------------------------------------------

    def record(self, tool_name: str, success: bool):

        now = time.time()

        if tool_name not in self.stats:
            self.stats[tool_name] = {
                "success": 0.0,
                "fail": 0.0,
            }
            self.last_update[tool_name] = now

        # Apply decay based on time difference
        delta = now - self.last_update[tool_name]
        decay_factor = self.DECAY ** delta

        self.stats[tool_name]["success"] *= decay_factor
        self.stats[tool_name]["fail"] *= decay_factor

        # Update counts
        if success:
            self.stats[tool_name]["success"] += 1
        else:
            self.stats[tool_name]["fail"] += 1

        self.last_update[tool_name] = now

    # ------------------------------------------------------------
    # Reliability Score
    # ------------------------------------------------------------

    def score(self, tool_name: str) -> float:

        s = self.stats.get(tool_name)

        if not s:
            return 0.5  # neutral

        total = s["success"] + s["fail"]

        if total < self.MIN_SAMPLES:
            # shrink toward neutral while learning
            base = s["success"] / total if total > 0 else 0.5
            return 0.5 + (base - 0.5) * (total / self.MIN_SAMPLES)

        return s["success"] / total

    # ------------------------------------------------------------
    # Volatility Indicator
    # ------------------------------------------------------------

    def volatility(self, tool_name: str) -> float:
        """
        Measures instability.
        High volatility means inconsistent performance.
        """
        s = self.stats.get(tool_name)
        if not s:
            return 0.0

        total = s["success"] + s["fail"]
        if total == 0:
            return 0.0

        p = s["success"] / total
        return 4 * p * (1 - p)  # maximum at p=0.5

    # ------------------------------------------------------------
    # Full Report
    # ------------------------------------------------------------

    def all_scores(self):
        return {
            t: {
                "reliability": self.score(t),
                "volatility": self.volatility(t),
            }
            for t in self.stats
        }
