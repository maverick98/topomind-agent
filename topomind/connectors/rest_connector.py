import requests
import logging
from typing import Dict, Any, Optional
from topomind.connectors.base import ExecutionConnector

logger = logging.getLogger(__name__)


class RestConnector(ExecutionConnector):
    """
    Generic REST execution connector.

    Responsible ONLY for transport.
    Does NOT perform validation.
    Does NOT perform model routing.
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

    # ============================================================
    # EXECUTION
    # ============================================================

    def execute(
        self,
        tool,
        arguments: Dict[str, Any],
        timeout: int = None,
    ) -> Dict[str, Any]:

        if not tool or not tool.name:
            raise RuntimeError("Invalid tool object passed to RestConnector")

        url = f"{self.base_url}/{tool.name}"
        effective_timeout = timeout or self.timeout_seconds

        logger.info("========== REST CONNECTOR ==========")
        logger.info(f"Tool: {tool.name}")
        logger.info(f"URL: {url}")
        logger.info(f"Payload: {arguments}")
        logger.info("====================================")

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

            # Raise if HTTP error
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"REST transport failure ({self.method} {url}): {e}"
            )

        # ------------------------------------------------------------
        # Response Parsing
        # ------------------------------------------------------------

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

        return data["output"]
