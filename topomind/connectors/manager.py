from typing import Dict
from .base import ExecutionConnector


class ConnectorManager:
    def __init__(self):
        self._connectors: Dict[str, ExecutionConnector] = {}

    def register(self, name: str, connector: ExecutionConnector):
        self._connectors[name] = connector

    def get(self, name: str) -> ExecutionConnector:
        return self._connectors[name]
