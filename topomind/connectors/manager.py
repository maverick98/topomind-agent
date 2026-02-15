from __future__ import annotations

from typing import Dict, Iterable
from threading import RLock
import logging

from .base import ExecutionConnector

logger = logging.getLogger(__name__)


class ConnectorManager:

    def __init__(self) -> None:
        self._connectors: Dict[str, ExecutionConnector] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, connector: ExecutionConnector) -> None:

        if not name or not isinstance(name, str):
            raise ValueError("Connector must have a valid string name.")

        if not isinstance(connector, ExecutionConnector):
            raise TypeError("Connector must implement ExecutionConnector.")

        with self._lock:
            if name in self._connectors:
                raise ValueError(f"Connector '{name}' already registered.")
            self._connectors[name] = connector

    def register_many(self, connectors: Dict[str, ExecutionConnector]) -> None:

        with self._lock:
            for name, connector in connectors.items():
                if name in self._connectors:
                    raise ValueError(f"Connector '{name}' already registered.")
                if not isinstance(connector, ExecutionConnector):
                    raise TypeError(f"{name} is not a valid ExecutionConnector.")

            self._connectors.update(connectors)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ExecutionConnector:

        with self._lock:
            try:
                return self._connectors[name]
            except KeyError:
                raise KeyError(f"Connector '{name}' is not registered.") from None

    def has_connector(self, name: str) -> bool:
        with self._lock:
            return name in self._connectors

    def list_connectors(self) -> Iterable[str]:
        with self._lock:
            return list(self._connectors.keys())

    # ------------------------------------------------------------------
    # Observability & Control
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, bool]:

        status = {}

        with self._lock:
            for name, conn in self._connectors.items():
                try:
                    status[name] = conn.health()
                except Exception as e:
                    logger.warning(
                        f"Health check failed for connector '{name}': {e}"
                    )
                    status[name] = False

        return status

    def shutdown_all(self) -> None:

        with self._lock:
            for name, conn in self._connectors.items():
                try:
                    conn.shutdown()
                except Exception as e:
                    logger.warning(
                        f"Shutdown failed for connector '{name}': {e}"
                    )

    def __len__(self) -> int:
        with self._lock:
            return len(self._connectors)
