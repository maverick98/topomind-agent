from topomind import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector
from topomind.connectors.ollama import OllamaConnector   

import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
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

# Echo tool (utility)
registry.register(
    Tool(
        name="echo",
        description="Echo back text",
        input_schema={"text": "string"},
        output_schema={"text": "string"},
        connector_name="local",
    )
)

#  REASON TOOL (THIS IS THE BRAIN)
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

print(agent.handle_query("What is Quantum Mechanics?"))
print(agent.handle_query("Who laid its foundations?"))
print(agent.handle_query("What is Bohr Einstein debate?"))

print("\n=== Conversation End ===\n")

# --------------------------------
# Inspect memory
# --------------------------------

print("--- Memory State ---")
for node in agent.memory.nodes():
    print(f"Type={node.type}, Turn={node.turn_created}, Value={node.value}")
