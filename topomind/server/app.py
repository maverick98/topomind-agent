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
from topomind.connectors.ollama import OllamaConnector
from topomind.connectors.groq import GroqConnector

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ============================================================
# PLANNER CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ============================================================

PLANNER_TYPE = "llm"
LLM_BACKEND = "groq"  # "ollama" or "groq"

if LLM_BACKEND == "ollama":
    PLANNER_MODEL = "phi3:mini"
elif LLM_BACKEND == "groq":
    PLANNER_MODEL = "llama-3.1-8b-instant"
else:
    raise ValueError(f"Unsupported LLM_BACKEND: {LLM_BACKEND}")

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
        # Local fallback connector
        self.connectors.register("local", FakeConnector())

        # Register LLM connector
        if LLM_BACKEND == "ollama":
            self.connectors.register(
                "llm",
                OllamaConnector(default_model=PLANNER_MODEL)
            )
        elif LLM_BACKEND == "groq":
            self.connectors.register(
                "llm",
                GroqConnector(model=PLANNER_MODEL)
            )

    def _build_agent(self):
        self.agent = TopoMindApp.create(
            planner_type=PLANNER_TYPE,
            model=PLANNER_MODEL,
            llm_backend=LLM_BACKEND,
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

app = FastAPI(title="TopoMind Dynamic Platform", version="5.2")

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
    execution_model: Optional[str] = ""


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
        "llm_backend": LLM_BACKEND,
    }


# ============================================================
# Capabilities
# ============================================================

@app.get("/capabilities")
def capabilities():
    tools = manager.registry.list_tools()

    return {
        "tool_count": len(tools),
        "tools": [
            {
                "name": t.name,
                "connector": t.connector_name,
                "strict": t.strict,
                "execution_model": t.execution_model,
                "version": t.version,
                "timeout_seconds": t.timeout_seconds,
                "retryable": t.retryable,
                "side_effect": t.side_effect,
                "tags": t.tags,
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

        # --------------------------------------------------
        # If agent returned dict (failure path / legacy)
        # --------------------------------------------------
        if isinstance(result, dict):
            return QueryResponse(
                status=result.get("status", "failure"),
                tool=result.get("tool_name"),
                output=result.get("output"),
                error=result.get("error"),
            )

        # --------------------------------------------------
        # Normal structured object
        # --------------------------------------------------
        return QueryResponse(
            status=getattr(result, "status", "failure"),
            tool=getattr(result, "tool_name", None),
            output=getattr(result, "output", None),
            error=getattr(result, "error", None),
        )

    except Exception as e:
        logging.exception("Query execution failed")
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================
# Register Tool
# ============================================================

@app.post("/register-tool")
def register_tool(request: ToolRegistrationRequest):
    try:
        tool = Tool(
            name=request.name,
            description=request.description,
            input_schema=request.input_schema,
            output_schema=request.output_schema or {},
            connector_name=request.connector,
            prompt=request.prompt or "",
            strict=request.strict or False,
            execution_model=request.execution_model or "",
        )

        manager.register_tool(tool)

        return {
            "status": "registered",
            "tool": request.name,
        }

    except Exception as e:
        logging.exception("Tool registration failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Register Connector
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
        logging.exception("Connector registration failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Clear Tools
# ============================================================

@app.post("/clear-tools")
def clear_tools():
    manager.clear_tools()
    return {"status": "all tools cleared"}
