from topomind import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector
from topomind.connectors.ollama import OllamaConnector

# NEW
from topomind.builtin.analytics import register_builtin_analytics

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# --------------------------------
# Consumer infrastructure
# --------------------------------

connectors = ConnectorManager()

# Local dummy tool connector
connectors.register("local", FakeConnector())

# LLM reasoning connector
connectors.register("llm", OllamaConnector(model="mistral"))

registry = ToolRegistry()

# --------------------------------
# Register Built-in Analytics
# --------------------------------

register_builtin_analytics(
    connectors=connectors,
    registry=registry,
)

# --------------------------------
# Custom Tools
# --------------------------------

# Echo tool
registry.register(
    Tool(
        name="echo",
        description="Echo back text",
        input_schema={"text": "string"},
        output_schema={"text": "string"},
        connector_name="local",
    )
)

# REASON TOOL (LLM brain)
registry.register(
    Tool(
        name="reason",
        description="Answer conceptual and knowledge questions",
        input_schema={"question": "string"},
        output_schema={"answer": "string"},
        connector_name="llm",
    )
)

# --------------------------------
# Create Agent
# --------------------------------

agent = TopoMindApp.create(
    planner_type="ollama",
    model="mistral",
    connectors=connectors,
    registry=registry,
)

# --------------------------------
# Run conversation
# --------------------------------

print("\n=== Conversation Start ===\n")

# ---- Arithmetic ----
print(agent.handle_query("What is 25 * 48 + 10?"))
print(agent.handle_query("Calculate math.sqrt(144) + 5"))
print(agent.handle_query("What is 7 squared?"))

# ---- Descriptive Stats ----
print(agent.handle_query("Find MEAN of [10, 20, 30, 40]"))
print(agent.handle_query("Find STD_DEV_SAMPLE of [10, 20, 30, 40]"))
print(agent.handle_query("Find VARIANCE of [10, 20, 30, 40]"))
print(agent.handle_query("Find MEDIAN of [10, 20, 30, 40]"))

# ---- Normalization ----
print(agent.handle_query("Compute Z_SCORE of [10, 20, 30, 40]"))
print(agent.handle_query("Compute COEFFICIENT_OF_VARIATION of [10, 20, 30, 40]"))

# ---- Relationship ----
print(agent.handle_query("Find COVARIANCE between [1,2,3] and [2,4,6]"))
print(agent.handle_query("Find CORRELATION between [1,2,3] and [2,4,6]"))

# ---- Regression ----
print(agent.handle_query("Find TREND_SLOPE for x=[1,2,3,4] and y=[2,4,6,8]"))
print(agent.handle_query("Find REGRESSION_INTERCEPT for x=[1,2,3,4] and y=[2,4,6,8]"))
print(agent.handle_query("Find R_SQUARED for x=[1,2,3,4] and y=[2,4,6,8]"))
print(agent.handle_query("Find ADJUSTED_R_SQUARED for x=[1,2,3,4] and y=[2,4,6,8]"))

# ---- Time Series Diagnostics ----
print(agent.handle_query("Compute AUTOCORRELATION of [10,20,30,40,50]"))
print(agent.handle_query("Compute LJUNG_BOX of [10,20,30,40,50] with lag 1"))

# ---- Error Metrics ----
print(agent.handle_query("Compute RMSE for actual=[10,20,30] and predicted=[12,18,29]"))
print(agent.handle_query("Compute MAPE for actual=[10,20,30] and predicted=[12,18,29]"))

# ---- Data Preprocessing ----
print(agent.handle_query("Find CLEAN_START_INDEX of [None,None,5,6,7]"))

# ---- Anomaly Detection ----
print(agent.handle_query("Perform OUTLIER_DETECTION on [10, 12, 11, 200, 13]"))

print("\n=== Conversation End ===\n")


# --------------------------------
# Inspect memory
# --------------------------------

print("--- Memory State ---")
for node in agent.memory.nodes():
    print(f"Type={node.type}, Turn={node.turn_created}, Value={node.value}")
