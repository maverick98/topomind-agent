from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from fastapi.middleware.cors import CORSMiddleware

from topomind import TopoMindApp
from topomind.tools.registry import ToolRegistry
from topomind.tools.schema import Tool
from topomind.connectors.manager import ConnectorManager
from topomind.connectors.base import FakeConnector
from topomind.connectors.ollama import OllamaConnector
from topomind.builtin.analytics import register_builtin_analytics


# ============================================================
# Agent Manager (Holds Connectors + Registry + Agent)
# ============================================================

class AgentManager:
    def __init__(self):
        self.connectors = ConnectorManager()
        self.registry = ToolRegistry()
        self._initialize_core()
        self._build_agent()

    def _initialize_core(self):
        # Core connectors
        self.connectors.register("local", FakeConnector())
        self.connectors.register("llm", OllamaConnector(model="mistral"))

        # Built-in tools
        register_builtin_analytics(
            connectors=self.connectors,
            registry=self.registry,
        )

    def _build_agent(self):
        self.agent = TopoMindApp.create(
            planner_type="ollama",
            model="mistral",
            connectors=self.connectors,
            registry=self.registry,
        )

    def get_agent(self):
        return self.agent

    def register_tool(self, tool: Tool):
        self.registry.register(tool)

    def register_connector(self, name: str, connector: Any):
        self.connectors.register(name, connector)


# ============================================================
# Instantiate Manager (Singleton)
# ============================================================

manager = AgentManager()
agent = manager.get_agent()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="TopoMind API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠ for testing only
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
    connector: str


class ConnectorRegistrationRequest(BaseModel):
    name: str
    type: str  # "ollama"
    model: Optional[str] = None


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
    tools = manager.registry.list_tool_names()
    return {
        "tool_count": len(tools),
        "tools": tools,
    }


# ============================================================
# Query Endpoint
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


# ============================================================
# Register Tool Endpoint
# ============================================================

@app.post("/register-tool")
def register_tool(request: ToolRegistrationRequest):

    try:
        tool = Tool(
            name=request.name,
            description=request.description,
            input_schema=request.input_schema,
            connector=request.connector,
        )

        manager.register_tool(tool)

        return {
            "status": "registered",
            "tool": request.name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Register Connector Endpoint
# ============================================================

@app.post("/register-connector")
def register_connector(request: ConnectorRegistrationRequest):

    try:
        if request.type == "ollama":
            connector = OllamaConnector(
                model=request.model or "mistral"
            )
        else:
            raise ValueError(f"Unsupported connector type: {request.type}")

        manager.register_connector(request.name, connector)

        return {
            "status": "registered",
            "connector": request.name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
