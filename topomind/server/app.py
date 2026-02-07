from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from topomind import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector
from topomind.connectors.ollama import OllamaConnector
from topomind.builtin.analytics import register_builtin_analytics

from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# Agent Factory
# ============================================================

def build_agent(
    extra_connectors: Optional[Dict[str, Any]] = None,
    extra_tools: Optional[List[Tool]] = None,
) -> TopoMindApp:
    """
    Builds TopoMind agent with optional external connectors and tools.
    """

    connectors = ConnectorManager()

    # Core connectors
    connectors.register("local", FakeConnector())
    connectors.register("llm", OllamaConnector(model="mistral"))

    # Consumer-provided connectors
    if extra_connectors:
        for name, connector in extra_connectors.items():
            connectors.register(name, connector)

    registry = ToolRegistry()

    # Built-in analytics
    register_builtin_analytics(connectors=connectors, registry=registry)

    # Consumer-provided tools
    if extra_tools:
        registry.register_many(extra_tools)

    return TopoMindApp.create(
        planner_type="ollama",
        model="mistral",
        connectors=connectors,
        registry=registry,
    )


# ============================================================
# Create Agent Instance (Singleton)
# ============================================================

agent = build_agent()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="TopoMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Request / Response Models
# ============================================================

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    status: str
    tool: Optional[str]
    output: Optional[Any]
    error: Optional[str]


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# Capabilities Endpoint
# ============================================================

@app.get("/capabilities")
def capabilities():
    tools = agent.registry.list_tool_names()
    return {
        "tool_count": len(tools),
        "tools": tools,
    }


# ============================================================
# Main Query Endpoint
# ============================================================

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):

    try:
        result = agent.handle_query(request.query)

        return QueryResponse(
            status=result.status,
            tool=result.tool_name,
            output=result.output,
            error=result.error,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
