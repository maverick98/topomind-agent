import logging
from ..planner.interface import ReasoningEngine
from ..planner.plan_model import Plan
from ..tools.executor import ToolExecutor
from ..memory.graph import MemoryGraph
from ..memory.updater import MemoryUpdater
from ..stability.signals import StabilitySignals
from ..models.observation import Observation
from .state import AgentState

logger = logging.getLogger(__name__)


class Agent:
    """
    TopoMind Agent Core

    Cognitive loop:
        Input → Memory → Stability → Planning → Execution → Memory
                        ↘ Session State ↙
    """

    def __init__(self, planner: ReasoningEngine, executor: ToolExecutor):
        self.planner = planner
        self.executor = executor

        self.memory = MemoryGraph()
        self.memory_updater = MemoryUpdater(self.memory)
        self.stability = StabilitySignals(self.memory)

        # Read-only access
        self.registry = executor.registry

        self.state = AgentState()

    def handle_query(self, user_input: str):
        logger.info(f"New turn: {user_input}")

        # --- Session ---
        self.state.new_turn(user_input)

        # --- User Observation ---
        user_obs = Observation(
            source="user",
            type="entity",
            payload=user_input,
            metadata={}
        )
        self.memory_updater.update_from_observation(user_obs)

        # --- Stability ---
        signals = self.stability.extract()
        logger.debug(f"Signals: {signals}")

        # --- Planning ---
        tools = self.registry.list_tools()
        plan: Plan = self.planner.generate_plan(user_input, signals, tools)
        self.state.record_plan(plan)

        if plan.is_empty():
            self.state.record_failure("Empty plan")
            return {"error": "Planner produced no action"}

        step = plan.first_step
        if not step.action:
            return {"error": "Invalid plan step"}

        self.state.last_tool_call = step.action

        # --- Execution ---
        logger.info(f"Executing tool: {step.action.tool_name}")
        result = self.executor.execute(step.action.tool_name, step.action.arguments)
        self.state.record_execution(step.action, result)

        # --- Store Result ---
        tool_obs = Observation(
            source="tool",
            type="result",
            payload=result,
            metadata={}
        )
        self.memory_updater.update_from_observation(tool_obs)

        logger.debug(f"Execution result: {result}")
        return result
