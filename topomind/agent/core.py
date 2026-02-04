import logging
import time
from ..planner.interface import ReasoningEngine
from ..planner.plan_model import Plan
from ..tools.executor import ToolExecutor
from ..memory.graph import MemoryGraph
from ..memory.updater import MemoryUpdater
from ..stability.signals import StabilitySignals
from ..models.observation import Observation
from .state import AgentState
from ..memory.observation_builder import ObservationBuilder
import pdb  

logger = logging.getLogger(__name__)


class Agent:

    def __init__(self, planner: ReasoningEngine, executor: ToolExecutor):
        self.planner = planner
        self.executor = executor

        self.memory = MemoryGraph()
        self.memory_updater = MemoryUpdater(self.memory)
        self.stability = StabilitySignals(self.memory)

        self.registry = executor.registry
        self.state = AgentState()
        self.obs_builder = ObservationBuilder()

    def handle_query(self, user_input: str):

        total_start = time.time()
        logger.info(f"[AGENT] New turn: {user_input}")

        # --- Session ---
        self.state.new_turn(user_input)

        # --- User Observation ---
        user_obs = Observation(source="user", type="entity", payload=user_input, metadata={})
        self.memory_updater.update_from_observation(user_obs)

        # --- Stability ---
        t0 = time.time()
        signals = self.stability.extract()
        logger.info(f"[STABILITY] {time.time() - t0:.2f}s")

        # --- Planning ---
        t0 = time.time()
        tools = self.registry.list_tools()
        plan: Plan = self.planner.generate_plan(user_input, signals, tools)
        logger.info(f"[PLANNER] {time.time() - t0:.2f}s")

        self.state.record_plan(plan)

        if plan.is_empty():
            logger.warning("[PLANNER] Empty plan produced")
            return {"error": "Planner produced no action"}

        step = plan.first_step
        if not step.action:
            logger.warning("[PLANNER] Invalid plan step")
            return {"error": "Invalid plan step"}

        self.state.last_tool_call = step.action

        # --- Execution ---
        logger.info(f"[EXECUTOR] Calling tool: {step.action.tool_name}")
        #pdb.set_trace()
        t0 = time.time()
        result = self.executor.execute(step.action.tool_name, step.action.arguments)
        logger.info(f"[EXECUTOR] {time.time() - t0:.2f}s")

        self.state.record_execution(step.action, result)

        # --- Store Raw Result ---
        tool_obs = Observation(source="tool", type="result", payload=result, metadata={})
        self.memory_updater.update_from_observation(tool_obs)

        # --- Semantic Knowledge Encoding ---
        if (
            result.tool_name == "reason"
            and getattr(result, "status", None) == "success"
            and isinstance(getattr(result, "output", None), dict)
            and "answer" in result.output
        ):
            logger.info("[SEMANTIC] Extracting structured knowledge")
            t0 = time.time()
            answer_text = result.output["answer"]
            semantic_observations = self.obs_builder.from_reason_result(answer_text)
            logger.info(f"[SEMANTIC] {time.time() - t0:.2f}s")

            for obs in semantic_observations:
                self.memory_updater.update_from_observation(obs)

        logger.info(f"[TOTAL TURN] {time.time() - total_start:.2f}s")
        logger.debug(f"Execution result: {result}")

        return result
