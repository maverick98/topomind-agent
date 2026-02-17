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
from topomind.connectors.cohere import CohereConnector

import logging

# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("topomind.server")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ============================================================
# PLANNER CONFIGURATION
# ============================================================

PLANNER_TYPE = "llm"
LLM_BACKEND = "cohere"  # "ollama" or "groq" or "cohere"

if LLM_BACKEND == "ollama":
    PLANNER_MODEL = "phi3:mini"
elif LLM_BACKEND == "groq":
    PLANNER_MODEL = "llama-3.1-8b-instant"
elif LLM_BACKEND == "cohere":
    PLANNER_MODEL = "command-a-03-2025"
else:
    raise ValueError(f"Unsupported LLM_BACKEND: {LLM_BACKEND}")

# ============================================================
# Agent Manager
# ============================================================

class AgentManager:

    def __init__(self):
        logger.info("[AGENT MANAGER] Initializing...")
        self.connectors = ConnectorManager()
        self.registry = ToolRegistry()
        self._initialize_core()
        self._build_agent()
        logger.info("[AGENT MANAGER] Ready")

    def _initialize_core(self):
        logger.info("[AGENT MANAGER] Registering connectors")

        self.connectors.register("local", FakeConnector())

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
        elif LLM_BACKEND == "cohere":
            self.connectors.register(
                "llm",
                CohereConnector(model=PLANNER_MODEL)
            )

        logger.info(
            "[AGENT MANAGER] LLM connector registered | backend=%s | model=%s",
            LLM_BACKEND,
            PLANNER_MODEL
        )

    def _build_agent(self):
        logger.info(
            "[AGENT MANAGER] Building agent | tool_count=%d",
            len(self.registry)
        )

        self.agent = TopoMindApp.create(
            planner_type=PLANNER_TYPE,
            model=PLANNER_MODEL,
            llm_backend=LLM_BACKEND,
            connectors=self.connectors,
            registry=self.registry,
        )

    def rebuild(self):
        logger.info("[AGENT MANAGER] Rebuilding agent")
        self._build_agent()

    def get_agent(self):
        return self.agent

    def register_tool(self, tool: Tool):
        logger.info(
            "[AGENT MANAGER] Register tool | name=%s | strict=%s | model=%s",
            tool.name,
            tool.strict,
            tool.execution_model
        )
        self.registry.register(tool)
        self.rebuild()

    def register_connector(self, name: str, connector: Any):
        logger.info(
            "[AGENT MANAGER] Register connector | name=%s",
            name
        )
        self.connectors.register(name, connector)
        self.rebuild()

    def clear_tools(self):
        logger.warning("[AGENT MANAGER] Clearing ALL tools")
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

    logger.info(
        "[CAPABILITIES] tool_count=%d | names=%s",
        len(tools),
        [t.name for t in tools]
    )

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
        logger.info(
            "[QUERY] text='%s' | tools=%s",
            request.query,
            manager.registry.list_tool_names()
        )

        result = manager.get_agent().handle_query(request.query)

        if isinstance(result, dict):
            logger.info("[QUERY] Legacy result returned")
            return QueryResponse(
                status=result.get("status", "failure"),
                tool=result.get("tool_name"),
                output=result.get("output"),
                error=result.get("error"),
            )

        logger.info(
            "[QUERY] Completed | status=%s | tool=%s",
            getattr(result, "status", "failure"),
            getattr(result, "tool_name", None)
        )

        return QueryResponse(
            status=getattr(result, "status", "failure"),
            tool=getattr(result, "tool_name", None),
            output=getattr(result, "output", None),
            error=getattr(result, "error", None),
        )

    except Exception:
        logger.exception("[QUERY] Execution failed")
        raise HTTPException(status_code=500, detail="Internal error")

# ============================================================
# Register Tool
# ============================================================

@app.post("/register-tool")
def register_tool(request: ToolRegistrationRequest):
    try:
        logger.info(
            "[REGISTER TOOL] name=%s | connector=%s | strict=%s | model=%s",
            request.name,
            request.connector,
            request.strict,
            request.execution_model
        )

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

        logger.info(
            "[REGISTER TOOL] Success | total_tools=%d",
            len(manager.registry)
        )

        return {"status": "registered", "tool": request.name}

    except Exception:
        logger.exception("[REGISTER TOOL] Failed")
        raise HTTPException(status_code=500, detail="Tool registration failed")

# ============================================================
# Register Connector
# ============================================================

@app.post("/register-connector")
def register_connector(request: ConnectorRegistrationRequest):
    try:
        logger.info(
            "[REGISTER CONNECTOR] name=%s | type=%s",
            request.name,
            request.type
        )

        connector = create_connector(request)
        manager.register_connector(request.name, connector)

        return {"status": "registered", "connector": request.name}

    except Exception:
        logger.exception("[REGISTER CONNECTOR] Failed")
        raise HTTPException(status_code=500, detail="Connector registration failed")

# ============================================================
# Clear Tools
# ============================================================

@app.post("/clear-tools")
def clear_tools():
    logger.warning("[CLEAR TOOLS] Clearing all tools")
    manager.clear_tools()
    return {"status": "all tools cleared"}
