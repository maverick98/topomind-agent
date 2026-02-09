import requests
from typing import Dict, Any, Optional
from topomind.connectors.base import ExecutionConnector


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

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: int = None,
    ) -> Dict[str, Any]:

        url = f"{self.base_url}/{tool_name}"
        effective_timeout = timeout or self.timeout_seconds

        try:
            if self.method == "POST":
                response = requests.post(
                    url,
                    json=arguments,
                    headers=self.headers,
                    timeout=effective_timeout,
                )
            elif self.method == "GET":
                response = requests.get(
                    url,
                    params=arguments,
                    headers=self.headers,
                    timeout=effective_timeout,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")

            response.raise_for_status()

            try:
                data = response.json()
            except ValueError:
                raise RuntimeError(
                    f"Remote service did not return valid JSON. "
                    f"Response text: {response.text}"
                )

            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Invalid response format from remote tool: {data}"
                )

            if "output" not in data:
                raise RuntimeError(
                    f"Missing 'output' field in remote response: {data}"
                )

            #  CRITICAL: Return only raw output for OutputValidator
            return data["output"]

        except Exception as e:
            raise RuntimeError(f"REST call failed: {str(e)}")
