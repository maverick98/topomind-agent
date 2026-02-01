from topomind.agent.core import Agent
from topomind.tools.executor import ToolExecutor
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.base import FakeConnector
from topomind.connectors.manager import ConnectorManager
from topomind.config import AgentConfig
from topomind.planner.factory import create_planner


# ------------------------------------------------
# Choose planner mode here
# ------------------------------------------------
PLANNER_MODE = "ollama"   # "rule" or "ollama"
OLLAMA_MODEL = "mistral"


# -------------------------------
# Register connectors
# -------------------------------
connectors = ConnectorManager()
connectors.register("local", FakeConnector())

# -------------------------------
# Register tools
# -------------------------------
registry = ToolRegistry()
registry.register(Tool(
    name="echo",
    description="Echo back text",
    input_schema={"text": "string"},
    output_schema={"text": "string"},
    connector_name="local"
))

# -------------------------------
# Planner configuration
# -------------------------------
if PLANNER_MODE == "ollama":
    config = AgentConfig(planner_type="ollama", model=OLLAMA_MODEL)
else:
    config = AgentConfig(planner_type="rule")

planner = create_planner(config)

# -------------------------------
# Setup Agent
# -------------------------------
executor = ToolExecutor(registry, connectors)
agent = Agent(planner, executor)

# -------------------------------
# Multi-turn interaction
# -------------------------------
print("\n=== Conversation Start ===\n")

print(agent.handle_query("What is Quantum Mechanics?"))
print(agent.handle_query("Who laid its foundations?"))
print(agent.handle_query("What is Bohr Einstein debate?"))

print("\n=== Conversation End ===")

# -------------------------------
# Inspect memory
# -------------------------------
print("\n--- Memory State ---")
for node in agent.memory.nodes():
    print(f"Type: {node.type}, Value: {node.value}, Turn: {node.turn_created}")
