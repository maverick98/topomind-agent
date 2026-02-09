from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

from topomind import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector
from topomind.connectors.ollama import OllamaConnector
from topomind.connectors.rest_connector import RestConnector

import logging

logging.basicConfig(level=logging.INFO)


# ============================================================
# Agent Manager (Fully Dynamic, No Hardcoding)
# ============================================================

class AgentManager:

    def __init__(self):
        self.connectors = ConnectorManager()
        self.registry = ToolRegistry()
        self._initialize_core()
        self._build_agent()

    def _initialize_core(self):
        # Base infrastructure connectors only
        self.connectors.register("local", FakeConnector())
        self.connectors.register("llm", OllamaConnector(model="phi3:mini"))

    def _build_agent(self):
        self.agent = TopoMindApp.create(
            planner_type="ollama",
            model="phi3:mini",
            connectors=self.connectors,
            registry=self.registry,
        )

    def rebuild(self):
        self._build_agent()

    def get_agent(self):
        return self.agent

    def register_tool(self, tool: Tool):
        self.registry.register(tool)
        self.rebuild()

    def register_connector(self, name: str, connector: Any):
        self.connectors.register(name, connector)
        self.rebuild()

    def clear_tools(self):
        self.registry = ToolRegistry()
        self.rebuild()


# ============================================================
# Instantiate Manager
# ============================================================

manager = AgentManager()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="TopoMind Dynamic Platform", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class ToolRegistrationRequest(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    connector: str
    prompt: Optional[str] = None
    strict: Optional[bool] = False


class ConnectorRegistrationRequest(BaseModel):
    name: str
    type: str  # "ollama" | "fake" | "rest"
    model: Optional[str] = None
    base_url: Optional[str] = None
    method: Optional[str] = "POST"
    timeout_seconds: Optional[int] = 10


# ============================================================
# Connector Factory (Clean Pattern)
# ============================================================

def create_connector(request: ConnectorRegistrationRequest):

    if request.type == "ollama":
        return OllamaConnector(
            model=request.model or "phi3:mini"
        )

    if request.type == "fake":
        return FakeConnector()

    if request.type == "rest":
        if not request.base_url:
            raise ValueError("base_url is required for rest connector")

        return RestConnector(
            base_url=request.base_url,
            method=request.method or "POST",
            timeout_seconds=request.timeout_seconds or 10,
        )

    raise ValueError(f"Unsupported connector type: {request.type}")


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# Capabilities
# ============================================================

@app.get("/capabilities")
def capabilities():
    tools = manager.registry.list_tool_names()
    return {
        "tool_count": len(tools),
        "tools": tools,
    }


# ============================================================
# Query
# ============================================================

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        result = manager.get_agent().handle_query(request.query)

        return QueryResponse(
            status=result.status,
            tool=result.tool_name,
            output=result.output,
            error=result.error,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Register Tool (Fully Dynamic)
# ============================================================

@app.post("/register-tool")
def register_tool(request: ToolRegistrationRequest):
    try:

        tool = Tool(
            name=request.name,
            description=request.description,
            input_schema=request.input_schema,
            output_schema=request.output_schema,
            connector_name=request.connector,
            prompt=request.prompt,
            strict=request.strict,
        )

        manager.register_tool(tool)

        return {
            "status": "registered",
            "tool": request.name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Register Connector (Dynamic)
# ============================================================

@app.post("/register-connector")
def register_connector(request: ConnectorRegistrationRequest):
    try:

        connector = create_connector(request)
        manager.register_connector(request.name, connector)

        return {
            "status": "registered",
            "connector": request.name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Clear Tools
# ============================================================

@app.post("/clear-tools")
def clear_tools():
    manager.clear_tools()
    return {"status": "all tools cleared"}
