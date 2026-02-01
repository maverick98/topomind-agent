from ..planner.interface import ReasoningEngine
from ..planner.plan_model import Plan
from ..tools.executor import ToolExecutor
from ..memory.graph import MemoryGraph
from ..memory.updater import MemoryUpdater
from ..stability.signals import StabilitySignals
from ..models.observation import Observation
from .state import AgentState


class Agent:
    """
    TopoMind Agent Core

    Orchestrates the full cognitive loop:

        Input → Memory → Stability → Planning → Execution → Memory
                        ↘ Session State ↙
    """

    def __init__(self, planner: ReasoningEngine, executor: ToolExecutor):
        self.planner = planner
        self.executor = executor

        # Long-term structured memory
        self.memory = MemoryGraph()
        self.memory_updater = MemoryUpdater(self.memory)

        # Stability analysis
        self.stability = StabilitySignals(self.memory)

        # Tool registry
        self.registry = executor.registry

        # Short-term session state
        self.state = AgentState()

    # ------------------------------------------------------------------
    # Main Interaction Loop
    # ------------------------------------------------------------------

    def handle_query(self, user_input: str):
        """
        Process a single user turn through the cognitive pipeline.
        """

        # --- 0. Session bookkeeping ---
        self.state.new_turn(user_input)

        # --- 1. Store user input as observation ---
        user_obs = Observation(
                                source="user",
                                type="entity",
                                payload=user_input,
                                metadata={}
                            )

        self.memory_updater.update_from_observation(user_obs)

        # --- 2. Extract stability signals ---
        signals = self.stability.extract()

        # --- 3. Planner receives available tools ---
        tools = self.registry.list_tools()

        # --- 4. Generate structured plan ---
        plan: Plan = self.planner.generate_plan(user_input, signals, tools)
        self.state.record_plan(plan)

        if plan.is_empty():
            return {"error": "Planner produced no action"}

        step = plan.first_step
        self.state.last_tool_call = step.action

        # --- 5. Execute tool ---
        result = self.executor.execute(
                            step.action.tool_name,
                            step.action.arguments
        )

        self.state.record_execution(step.action, result)

        # --- 6. Store tool result in memory ---
        tool_obs = Observation(
                                source="tool",
                                type="result",
                                payload=result,
                                metadata={}
        )

        self.memory_updater.update_from_observation(tool_obs)

        return result
