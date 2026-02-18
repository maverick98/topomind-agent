from fastapi import FastAPI, HTTPException, Response, status
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
LLM_BACKEND = "cohere"

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
        self.connectors = ConnectorManager()  # Persistent
        self.registry = ToolRegistry()
        self._initialize_core()
        self._build_agent()
        logger.info("[AGENT MANAGER] Ready")

    def _initialize_core(self):
        logger.info("[AGENT MANAGER] Registering core connectors")

        # Core connectors are always strict (non-persistent)
        if not self.connectors.is_registered("local"):
            self.connectors.register("local", FakeConnector())

        if LLM_BACKEND == "ollama":
            if not self.connectors.is_registered("llm"):
                self.connectors.register(
                    "llm",
                    OllamaConnector(default_model=PLANNER_MODEL)
                )
        elif LLM_BACKEND == "groq":
            if not self.connectors.is_registered("llm"):
                self.connectors.register(
                    "llm",
                    GroqConnector(model=PLANNER_MODEL)
                )
        elif LLM_BACKEND == "cohere":
            if not self.connectors.is_registered("llm"):
                self.connectors.register(
                    "llm",
                    CohereConnector(model=PLANNER_MODEL)
                )

        logger.info(
            "[AGENT MANAGER] LLM connector ready | backend=%s | model=%s",
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

    def register_tool(self, tool: Tool) -> str:
        logger.info(
            "[AGENT MANAGER] Register tool | name=%s",
            tool.name,
        )

        result = self.registry.register_or_update(tool)

        # Always rebuild to ensure planner reflects latest contracts
        self.rebuild()

        return result


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

app = FastAPI(title="TopoMind Dynamic Platform", version="5.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Models
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
# Query
# ============================================================

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        result = manager.get_agent().handle_query(request.query)

        if isinstance(result, dict):
            return QueryResponse(
                status=result.get("status", "failure"),
                tool=result.get("tool_name"),
                output=result.get("output"),
                error=result.get("error"),
            )

        return QueryResponse(
            status=getattr(result, "status", "failure"),
            tool=getattr(result, "tool_name", None),
            output=getattr(result, "output", None),
            error=getattr(result, "error", None),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception:
        logger.exception("[QUERY] Execution failed")
        raise HTTPException(status_code=500, detail="Internal error")


# ============================================================
# Register Tool
# ============================================================

@app.post("/register-tool")
def register_tool(request: ToolRegistrationRequest, response: Response):
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

        result = manager.register_tool(tool)

        if result == "registered":
            response.status_code = status.HTTP_201_CREATED
        else:
            response.status_code = status.HTTP_200_OK

        return {
            "status": result,
            "tool": request.name
        }

    except Exception:
        logger.exception("[REGISTER TOOL] Failed")
        raise HTTPException(status_code=500, detail="Tool registration failed")



# ============================================================
# Register Connector (Persistent + Idempotent)
# ============================================================

@app.post("/register-connector")
def register_connector(
    request: ConnectorRegistrationRequest,
    response: Response,
):
    try:
        connector = create_connector(request)

        result = manager.connectors.register_or_update(
            request.name,
            connector,
            metadata=request.dict(),  # REQUIRED FOR PERSISTENCE
        )

        if result == "registered":
            response.status_code = status.HTTP_201_CREATED

        return {
            "status": result,
            "connector": request.name
        }

    except Exception as e:
        logger.exception("[REGISTER CONNECTOR] Failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Connector Lifecycle
# ============================================================

@app.get("/connectors")
def list_connectors():
    return {
        "count": len(manager.connectors),
        "connectors": manager.connectors.list_connectors(),
    }


@app.post("/undeploy-connector/{name}")
def undeploy_connector(name: str):
    try:
        manager.connectors.undeploy(name)
        return {"status": "undeployed", "connector": name}
    except KeyError:
        raise HTTPException(status_code=404, detail="Connector not found")


@app.post("/deploy-connector/{name}")
def deploy_connector(name: str):
    try:
        manager.connectors.deploy(name)
        return {"status": "deployed", "connector": name}
    except KeyError:
        raise HTTPException(status_code=404, detail="Connector not found")


# ============================================================
# Clear Tools
# ============================================================

@app.post("/clear-tools")
def clear_tools():
    manager.clear_tools()
    return {"status": "all tools cleared"}
