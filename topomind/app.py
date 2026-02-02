from typing import Optional

from .agent.core import Agent
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry
from .connectors.manager import ConnectorManager
from .planner.factory import create_planner
from .config import AgentConfig


class TopoMindApp:
    """
    Top-level framework facade for constructing a TopoMind Agent.

    This class represents the **official public entry point** of the
    TopoMind framework. It hides internal wiring complexity and enforces
    a clean separation between:

        • Framework-owned cognition (agent, planner, memory, execution)
        • Consumer-owned infrastructure (tools and connectors)

    Design Principles
    -----------------
    • The framework MUST NOT assume any execution environment
    • Tools and connectors are always consumer-provided
    • The returned Agent is fully wired and ready to run
    • No side effects (no global state, no registrations)

    This mirrors mature framework design (e.g., FastAPI, SQLAlchemy),
    where the framework provides structure and guarantees, while the
    application provides domain-specific components.
    """

    @staticmethod
    def create(
        *,
        planner_type: str,
        model: Optional[str],
        connectors: ConnectorManager,
        registry: ToolRegistry,
    ) -> Agent:
        """
        Construct and return a fully initialized TopoMind Agent.

        This method performs **pure assembly only**:
        it does not mutate global state, register tools, or
        create connectors internally.

        Parameters
        ----------
        planner_type : str
            Identifier for the planner backend to use.

            Supported values typically include:
                • "rule"   – deterministic rule-based planner
                • "ollama" – local LLM planner via Ollama
                • "openai" – OpenAI-backed LLM planner

        model : Optional[str]
            Model identifier for LLM-based planners.
            Ignored for non-LLM planners.

            Examples:
                "mistral", "llama3", "gpt-4o-mini"

        connectors : ConnectorManager
            Consumer-provided execution backends.

            Connectors define *how* tools are executed
            (local code, APIs, databases, services, etc.).

        registry : ToolRegistry
            Consumer-provided tool definitions.

            Tools define *what* the agent is allowed to do.
            The framework never invents or assumes tools.

        Returns
        -------
        Agent
            A fully constructed TopoMind Agent instance,
            ready to process user input via `handle_query()`.

        Architectural Notes
        -------------------
        • Planner selection is delegated to `planner.factory`
        • Tool execution is guarded by ToolExecutor firewalls
        • Memory and stability layers are initialized inside Agent
        • No framework defaults leak into consumer infrastructure
        """

        # Planner configuration is framework-owned
        config = AgentConfig(
            planner_type=planner_type,
            model=model,
        )

        # Planner construction (LLM / rule-based / etc.)
        planner = create_planner(config)

        # Execution boundary (tools + connectors)
        executor = ToolExecutor(registry, connectors)

        # Final agent assembly
        return Agent(planner, executor)
