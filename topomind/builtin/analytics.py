"""
Built-in analytics registration helper.

This module wires together:
- Connectors (execution layer)
- Tool contracts (declarative layer)
- Optional schema registration (contract history)

Usage
-----
from topomind.builtin.analytics import register_builtin_analytics

register_builtin_analytics(
    connectors=connector_manager,
    registry=tool_registry,
    schema_registry=schema_registry,  # optional
)
"""

from topomind.connectors.math_connector import MathConnector
from topomind.connectors.statistics_connector import StatisticsConnector
from topomind.connectors.timeseries_connector import TimeSeriesConnector

from topomind.tools.builtin.math_tool import MATH_TOOL
from topomind.tools.builtin.statistics_tool import STATISTICS_TOOL
from topomind.tools.builtin.timeseries_tool import TIMESERIES_TOOL


def register_builtin_analytics(
    connectors,
    registry,
    schema_registry=None,
) -> None:
    """
    Register built-in analytical connectors and tools.

    Parameters
    ----------
    connectors : ConnectorManager
        The connector manager instance.

    registry : ToolRegistry
        The tool registry instance.

    schema_registry : SchemaRegistry, optional
        If provided, tool schemas will be registered for version tracking.
    """

    # -------------------------
    # Register Connectors
    # -------------------------

    connectors.register("math", MathConnector())
    connectors.register("statistics", StatisticsConnector())
    connectors.register("timeseries", TimeSeriesConnector())

    # -------------------------
    # Register Tools
    # -------------------------

    registry.register(MATH_TOOL)
    registry.register(STATISTICS_TOOL)
    registry.register(TIMESERIES_TOOL)

    # -------------------------
    # Optional Schema Tracking
    # -------------------------

    if schema_registry is not None:
        schema_registry.register(MATH_TOOL)
        schema_registry.register(STATISTICS_TOOL)
        schema_registry.register(TIMESERIES_TOOL)
