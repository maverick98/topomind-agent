from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

from topomind import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector
from topomind.connectors.ollama import OllamaConnector
from topomind.builtin.analytics import register_builtin_analytics


# --------------------------------
# Build Agent (same as standalone)
# --------------------------------

connectors = ConnectorManager()
connectors.register("local", FakeConnector())
connectors.register("llm", OllamaConnector(model="mistral"))

registry = ToolRegistry()
register_builtin_analytics(connectors=connectors, registry=registry)

agent = TopoMindApp.create(
    planner_type="ollama",
    model="mistral",
    connectors=connectors,
    registry=registry,
)

# --------------------------------
# FastAPI App
# --------------------------------

app = FastAPI(title="TopoMind API")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_endpoint(request: QueryRequest):

    result = agent.handle_query(request.query)

    return {
        "status": result.status,
        "tool": result.tool_name,
        "output": result.output,
        "error": result.error,
    }
