from typing import Optional

from topomind.agent.core import Agent
from topomind.tools.executor import ToolExecutor
from topomind.tools.registry import ToolRegistry
from topomind.connectors.manager import ConnectorManager
from topomind.planner.factory import create_planner
from topomind.config import AgentConfig


class TopoMindApp:
    """
    Server-owned application assembler.
    SINGLE source of truth.
    """

    @staticmethod
    def create(
        *,
        planner_type: str,
        model: Optional[str],
        connectors: ConnectorManager,
        registry: ToolRegistry,
    ) -> Agent:

        config = AgentConfig(
            planner_type=planner_type,
            model=model,
        )

        planner = create_planner(config)
        executor = ToolExecutor(registry, connectors)

        return Agent(planner, executor)
