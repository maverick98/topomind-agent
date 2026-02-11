from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

from topomind.server.app_core import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector
from topomind.connectors.rest_connector import RestConnector

import logging

logging.basicConfig(level=logging.INFO)

# ============================================================
# PLANNER CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ============================================================

PLANNER_TYPE = "ollama"
PLANNER_MODEL = "phi3:mini"

# ============================================================
# Agent Manager
# ============================================================

class AgentManager:

    def __init__(self):
        self.connectors = ConnectorManager()
        self.registry = ToolRegistry()
        self._initialize_core()
        self._build_agent()

    def _initialize_core(self):
        self.connectors.register("local", FakeConnector())

    def _build_agent(self):
        self.agent = TopoMindApp.create(
            planner_type=PLANNER_TYPE,
            model=PLANNER_MODEL,
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

app = FastAPI(title="TopoMind Dynamic Platform", version="5.1")

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
    execution_model: Optional[str] = ""   # ✅ NEW


class ConnectorRegistrationRequest(BaseModel):
    name: str
    type: str
    base_url: Optional[str] = None
    method: Optional[str] = "POST"
    timeout_seconds: Optional[int] = 10


# ============================================================
# Connector Factory
# ============================================================

def create_connector(request: ConnectorRegistrationRequest):

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
    return {
        "status": "ok",
        "planner_type": PLANNER_TYPE,
        "planner_model": PLANNER_MODEL,
    }


# ============================================================
# Capabilities (Enhanced)
# ============================================================

@app.get("/capabilities")
def capabilities():
    tools = manager.registry.get_all()

    return {
        "tool_count": len(tools),
        "tools": [
            {
                "name": t.name,
                "connector": t.connector_name,
                "strict": t.strict,
                "execution_model": t.execution_model,
            }
            for t in tools
        ]
    }


# ============================================================
# Query Endpoint
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
# Register Tool (Now Logs Everything)
# ============================================================

@app.post("/register-tool")
def register_tool(request: ToolRegistrationRequest):
    try:
        logging.info("========== TOOL REGISTRATION RECEIVED ==========")
        logging.info(request.json())
        logging.info("================================================")

        tool = Tool(
            name=request.name,
            description=request.description,
            input_schema=request.input_schema,
            output_schema=request.output_schema,
            connector_name=request.connector,
            prompt=request.prompt,
            strict=request.strict,
            execution_model=request.execution_model or "",
        )

        logging.info("========== TOOL OBJECT CREATED ==========")
        logging.info(f"name: {tool.name}")
        logging.info(f"connector: {tool.connector_name}")
        logging.info(f"strict: {tool.strict}")
        logging.info(f"execution_model: {tool.execution_model}")
        logging.info("==========================================")

        manager.register_tool(tool)

        return {
            "status": "registered",
            "tool": request.name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Register Connector
# ============================================================

@app.post("/register-connector")
def register_connector(request: ConnectorRegistrationRequest):
    try:
        connector = create_connector(request)
        manager.register_connector(request.name, connector)

        logging.info(f"Connector registered: {request.name}")

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
