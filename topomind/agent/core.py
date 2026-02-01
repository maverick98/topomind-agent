from ..planner.interface import ReasoningEngine
from ..tools.executor import ToolExecutor
from ..memory.graph import MemoryGraph
from ..memory.updater import MemoryUpdater
from ..stability.signals import StabilitySignals


class Agent:
    """
    TopoMind Agent Core

    Orchestrates:
    - Memory updates per turn
    - Stability signal extraction
    - Planning
    - Tool execution
    """

    def __init__(self, planner: ReasoningEngine, executor: ToolExecutor):
        self.planner = planner
        self.executor = executor

        # Structured memory graph
        self.memory = MemoryGraph()
        self.memory_updater = MemoryUpdater(self.memory)

        # Stability analysis layer
        self.stability = StabilitySignals(self.memory)

        # Access tool registry via executor
        self.registry = executor.registry

    def handle_query(self, user_input: str):
        """
        Process a single user turn.
        """

        # --- 1. Update memory with new user input ---
        self.memory_updater.update_from_input(user_input)

        # --- 2. Extract stability signals ---
        signals = self.stability.extract()
        print(f"[DEBUG] Stability Signals: {signals}")

        # --- 3. Provide tool list to planner ---
        tools = self.registry.list_tools()

        # --- 4. Generate plan ---
        plan = self.planner.generate_plan(user_input, signals, tools)

        # --- 5. Execute plan ---
        result = self.executor.execute(plan["tool"], plan["args"])

        # --- 6. Store result in memory ---
        self.memory.add_node("result", result)

        return result
