import requests
from typing import Dict, Any, Optional
from topomind.connectors.base import ExecutionConnector


class RestConnector(ExecutionConnector):
    """
    Generic REST connector.
    Allows TopoMind to call external HTTP services.
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

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        timeout: int,
    ) -> Any:

        url = f"{self.base_url}/{tool_name}"

        try:
            if self.method == "POST":
                response = requests.post(
                    url,
                    json=args,
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
            elif self.method == "GET":
                response = requests.get(
                    url,
                    params=args,
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")

            response.raise_for_status()
            return response.json()

        except Exception as e:
            raise RuntimeError(f"REST call failed: {e}")
