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
        logger.info("====================================================")
        logger.info(f"[AGENT] New turn: {user_input}")
        logger.info("====================================================")

        self._start_turn(user_input)

        signals = self._extract_stability()
        tools = self.registry.list_tools()

        logger.info(f"[TOOLS AVAILABLE] {[t.name for t in tools]}")
        logger.debug(f"[TOOLS FULL OBJECTS] {tools}")
        logger.info(f"[STABILITY SIGNALS] {signals}")

        plan = self._plan(user_input, signals, tools)
        logger.debug(f"[RAW PLAN OBJECT] {plan}")

        if plan is None:
            return self._failure_response("Planner produced no action")

        result = self._execute_with_possible_replan(
            user_input, signals, tools, plan
        )

        logger.debug(f"[FINAL RESULT OBJECT BEFORE FORMAT] {result}")

        self._handle_semantic_encoding(result)

        logger.info(f"[TOTAL TURN] {time.time() - total_start:.2f}s")
        logger.debug(f"[Execution result object] {result}")

        return self._format_response(result)

    # ============================================================
    # TURN INITIALIZATION
    # ============================================================

    def _start_turn(self, user_input: str):
        self.state.new_turn(user_input)

        logger.debug("[MEMORY] Creating user observation")

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
        logger.debug(f"[STABILITY RAW SIGNALS] {signals}")
        return signals

    # ============================================================
    # PLANNING
    # ============================================================

    def _plan(self, user_input, signals, tools):

        t0 = time.time()
        plan: Plan = self.planner.generate_plan(user_input, signals, tools)
        logger.info(f"[PLANNER] {time.time() - t0:.2f}s")
        logger.debug(f"[PLAN RAW RETURN] {plan}")

        if not plan or plan.is_empty():
            logger.warning("[PLANNER] Empty plan produced")
            return None

        logger.info(f"[PLAN STEPS COUNT] {len(plan.steps)}")

        for step in plan.steps:
            logger.info(f"[PLAN STEP] Tool={step.action.tool_name} Args={step.action.arguments}")

        step = plan.first_step
        if not step or not step.action:
            logger.warning("[PLANNER] Invalid plan step")
            return None

        self.state.record_plan(plan)
        self.state.last_tool_call = step.action

        return plan

    # ============================================================
    # EXECUTION + SAFE CHAINING
    # ============================================================

    def _execute_with_possible_replan(self, user_input, signals, tools, plan):

        previous_result = None

        for step in plan:

            logger.info(f"[EXECUTION] Running step: {step.action.tool_name}")
            logger.debug(f"[ORIGINAL STEP ARGS] {step.action.arguments}")

            working_args = dict(step.action.arguments)
            logger.debug(f"[WORKING ARGS COPY] {working_args}")

            if previous_result and isinstance(previous_result.output, dict):
                if (
                    "code" in previous_result.output
                    and "code" not in working_args
                ):
                    working_args["code"] = previous_result.output["code"]
                    logger.debug("[CHAINING] Injected code from previous result")

            logger.debug(f"[FINAL ARGS SENT TO EXECUTOR] {working_args}")

            result = self._execute_step(step.action.tool_name, working_args)

            logger.debug(f"[STEP RESULT OBJECT] {result}")
            logger.debug(f"[STEP RESULT TYPE] {type(result)}")

            if getattr(result, "status", None) != "success":
                logger.warning("[EXECUTION] Step failed. Stopping chain.")
                logger.warning(f"[STEP FAILURE STATUS] {getattr(result, 'status', None)}")
                logger.warning(f"[STEP FAILURE ERROR] {getattr(result, 'error', None)}")
                return result

            previous_result = result

        return previous_result

    # ============================================================
    # SINGLE STEP EXECUTION
    # ============================================================

    def _execute_step(self, tool_name, args):

        logger.info(f"[EXECUTOR] Calling tool: {tool_name}")
        logger.debug(f"[EXECUTOR INPUT] Tool={tool_name}, Args={args}")

        t0 = time.time()
        result = self.executor.execute(tool_name, args)
        logger.info(f"[EXECUTOR] {time.time() - t0:.2f}s")

        logger.debug(f"[EXECUTOR RAW RESULT] {result}")
        logger.debug(f"[EXECUTOR RESULT TYPE] {type(result)}")

        self.state.record_execution(tool_name, result)

        success = getattr(result, "status", None) == "success"
        self.tool_reliability.record(tool_name, success)

        logger.debug(f"[RELIABILITY SCORE] {self.tool_reliability.all_scores()}")

        tool_obs = Observation(
            source="tool",
            type="result",
            payload=result,
            metadata={},
        )

        self.memory_updater.update_from_observation(tool_obs)

        return result

    # ============================================================
    # SEMANTIC ENCODING
    # ============================================================

    def _handle_semantic_encoding(self, result):

        if not result:
            logger.debug("[SEMANTIC] No result to encode")
            return

        if (
            getattr(result, "tool_name", None) == "reason"
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

    # ============================================================
    # RESPONSE FORMATTERS
    # ============================================================

    def _format_response(self, result):

        logger.debug(f"[FORMAT RESPONSE INPUT] {result}")
        logger.debug(f"[FORMAT RESPONSE TYPE] {type(result)}")

        if not result:
            return self._failure_response("No result produced")

        return {
            "status": getattr(result, "status", None),
            "tool": getattr(result, "tool_name", None),
            "output": getattr(result, "output", None),
            "error": getattr(result, "error", None),
        }

    def _failure_response(self, message: str):
        logger.error(f"[FAILURE RESPONSE] {message}")
        return {
            "status": "failure",
            "tool": None,
            "output": None,
            "error": message,
        }
