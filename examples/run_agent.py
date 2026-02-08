from topomind import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector
from topomind.connectors.ollama import OllamaConnector
from topomind.builtin.analytics import register_builtin_analytics

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def build_agent():
    connectors = ConnectorManager()

    connectors.register("local", FakeConnector())
    connectors.register("llm", OllamaConnector(model="mistral"))

    registry = ToolRegistry()

    register_builtin_analytics(
        connectors=connectors,
        registry=registry,
    )

    registry.register(
        Tool(
            name="echo",
            description="Echo back text",
            input_schema={"text": "string"},
            output_schema={"text": "string"},
            connector_name="local",
        )
    )

    registry.register(
        Tool(
            name="reason",
            description="Answer conceptual and knowledge questions",
            input_schema={"question": "string"},
            output_schema={"answer": "string"},
            connector_name="llm",
        )
    )

    return TopoMindApp.create(
        planner_type="ollama",
        model="mistral",
        connectors=connectors,
        registry=registry,
    )


if __name__ == "__main__":

    agent = build_agent()

    print("\n=== Conversation Start ===\n")

    queries = [
        "What is 25 * 48 + 10?",
        "Calculate math.sqrt(144) + 5",
        "What is 7 squared?",
        "Find MEAN of [10, 20, 30, 40]",
        "Find STD_DEV_SAMPLE of [10, 20, 30, 40]",
        "Find VARIANCE of [10, 20, 30, 40]",
        "Find MEDIAN of [10, 20, 30, 40]",
        "Compute Z_SCORE of [10, 20, 30, 40]",
        "Compute COEFFICIENT_OF_VARIATION of [10, 20, 30, 40]",
        "Find COVARIANCE between [1,2,3] and [2,4,6]",
        "Find CORRELATION between [1,2,3] and [2,4,6]",
        "Find TREND_SLOPE for x=[1,2,3,4] and y=[2,4,6,8]",
        "Find REGRESSION_INTERCEPT for x=[1,2,3,4] and y=[2,4,6,8]",
        "Find R_SQUARED for x=[1,2,3,4] and y=[2,4,6,8]",
        "Find ADJUSTED_R_SQUARED for x=[1,2,3,4] and y=[2,4,6,8]",
        "Compute AUTOCORRELATION of [10,20,30,40,50]",
        "Compute LJUNG_BOX of [10,20,30,40,50] with lag 1",
        "Compute RMSE for actual=[10,20,30] and predicted=[12,18,29]",
        "Compute MAPE for actual=[10,20,30] and predicted=[12,18,29]",
        "Find CLEAN_START_INDEX of [None,None,5,6,7]",
        "Perform OUTLIER_DETECTION on [10, 12, 11, 200, 13]",
    ]

    for q in queries:
        print(f"\n>>> {q}")
        print(agent.handle_query(q))

    print("\n=== Conversation End ===\n")

    print("--- Memory State ---")
    for node in agent.memory.nodes():
        print(f"Type={node.type}, Turn={node.turn_created}, Value={node.value}")
