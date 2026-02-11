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
from ..learning import ToolReliability

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

        self.tool_reliability = ToolReliability()

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    def handle_query(self, user_input: str):

        total_start = time.time()
        logger.info(f"[AGENT] New turn: {user_input}")

        self._start_turn(user_input)

        signals = self._extract_stability()

        tools = self.registry.list_tools()

        plan = self._plan(user_input, signals, tools)
        if plan is None:
            return {"error": "Planner produced no action"}

        result = self._execute_with_possible_replan(
            user_input, signals, tools, plan
        )

        self._handle_semantic_encoding(result)

        logger.info(f"[TOTAL TURN] {time.time() - total_start:.2f}s")
        logger.debug(f"Execution result: {result}")

        return result

    # ============================================================
    # TURN INITIALIZATION
    # ============================================================

    def _start_turn(self, user_input: str):
        self.state.new_turn(user_input)

        user_obs = Observation(
            source="user",
            type="entity",
            payload=user_input,
            metadata={},
        )
        self.memory_updater.update_from_observation(user_obs)

    # ============================================================
    # STABILITY
    # ============================================================

    def _extract_stability(self):
        t0 = time.time()
        signals = self.stability.extract()
        logger.info(f"[STABILITY] {time.time() - t0:.2f}s")
        return signals

    # ============================================================
    # PLANNING
    # ============================================================

    def _plan(self, user_input, signals, tools):

        t0 = time.time()
        plan: Plan = self.planner.generate_plan(user_input, signals, tools)
        logger.info(f"[PLANNER] {time.time() - t0:.2f}s")

        self.state.record_plan(plan)

        if plan.is_empty():
            logger.warning("[PLANNER] Empty plan produced")
            return None

        step = plan.first_step
        if not step.action:
            logger.warning("[PLANNER] Invalid plan step")
            return None

        self.state.last_tool_call = step.action
        return plan

    # ============================================================
    # EXECUTION + REPLAN
    # ============================================================

    def _execute_with_possible_replan(self, user_input, signals, tools, plan):

        previous_result = None

        for step in plan:

            logger.info(f"[EXECUTION] Running step: {step.action.tool_name}")

            # ----------------------------------------------------------
            # Automatic chaining between steps
            # ----------------------------------------------------------
            if previous_result and isinstance(previous_result.output, dict):
                if "code" in previous_result.output and "code" not in step.action.arguments:
                    step.action.arguments["code"] = previous_result.output["code"]

            result = self._execute_step(step)

            if getattr(result, "status", None) != "success":
                logger.warning("[EXECUTION] Step failed. Stopping execution chain.")
                return result

            previous_result = result

        return previous_result


    def _execute_step(self, step):

        logger.info(f"[EXECUTOR] Calling tool: {step.action.tool_name}")
        t0 = time.time()

        result = self.executor.execute(
            step.action.tool_name,
            step.action.arguments,
        )

        logger.info(f"[EXECUTOR] {time.time() - t0:.2f}s")

        self.state.record_execution(step.action, result)

        # Reliability tracking (unchanged behavior)
        success = getattr(result, "status", None) == "success"
        self.tool_reliability.record(step.action.tool_name, success)

        tool_obs = Observation(
            source="tool",
            type="result",
            payload=result,
            metadata={},
        )
        self.memory_updater.update_from_observation(tool_obs)

        return result

    def _should_replan(self, step, result):

        return (
            getattr(result, "status", None) != "success"
            or getattr(result, "stability_signal", 1.0) < 0.5
            or step.confidence < 0.3
        )

    # ============================================================
    # SEMANTIC ENCODING
    # ============================================================

    def _handle_semantic_encoding(self, result):

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
