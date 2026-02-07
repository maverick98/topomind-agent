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

print(agent.handle_query("What is 25 * 48 + 10?"))
print(agent.handle_query("Calculate math.sqrt(144) + 5"))
print(agent.handle_query("Repeat: Hello TopoMind"))
print(agent.handle_query("What is 7 squared?"))
print(agent.handle_query("Tell me about Einstein"))
print(agent.handle_query("What is 100 / 4 - 3?"))

# New built-in capabilities
print(agent.handle_query("Find mean of [10, 20, 30, 40]"))
print(agent.handle_query("Find correlation between [1,2,3] and [2,4,6]"))
print(agent.handle_query("Compute moving average of [10,20,30,40,50] with window 2"))

print("\n=== Conversation End ===\n")

# --------------------------------
# Inspect memory
# --------------------------------

print("--- Memory State ---")
for node in agent.memory.nodes():
    print(f"Type={node.type}, Turn={node.turn_created}, Value={node.value}")
