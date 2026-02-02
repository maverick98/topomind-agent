from topomind import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector


# --------------------------------
# Consumer infrastructure
# --------------------------------

connectors = ConnectorManager()
connectors.register("local", FakeConnector())

registry = ToolRegistry()
registry.register(
    Tool(
        name="echo",
        description="Echo back text",
        input_schema={"text": "string"},
        output_schema={"text": "string"},
        connector_name="local",
    )
)

# --------------------------------
# Create Agent (ONE LINE 🎯)
# --------------------------------

agent = TopoMindApp.create(
    planner_type="ollama",     # or "rule"
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
