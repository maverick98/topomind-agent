import requests
from typing import Dict, Any, Optional
from topomind.connectors.base import ExecutionConnector
from topomind.models.tool_result import ToolResult


class RestConnector(ExecutionConnector):
    """
    Generic REST execution connector.
    Allows TopoMind to invoke external HTTP services.
    """

    def __init__(
        self,
        base_url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.method = method.upper()
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

    def execute(self, tool, arguments: Dict[str, Any], **kwargs) -> ToolResult:

        url = f"{self.base_url}/{tool.name}"

        try:
            if self.method == "POST":
                response = requests.post(
                    url,
                    json=arguments,
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
            elif self.method == "GET":
                response = requests.get(
                    url,
                    params=arguments,
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")

            response.raise_for_status()
            data = response.json()

            return ToolResult(
                tool_name=tool.name,
                tool_version=tool.version,
                status=data.get("status", "success"),
                output=data.get("output"),
                error=data.get("error"),
                latency_ms=0,
                stability_signal=1.0,
            )

        except Exception as e:
            return ToolResult(
                tool_name=tool.name,
                tool_version=tool.version,
                status="failure",
                output=None,
                error=f"REST call failed: {str(e)}",
                latency_ms=0,
                stability_signal=0.0,
            )
