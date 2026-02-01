from __future__ import annotations

from typing import Dict, Iterable
from threading import RLock

from .base import ExecutionConnector


class ConnectorManager:
    """
    Registry and control layer for execution connectors.

    This component is the final boundary between the agent runtime
    and external systems. It manages connector registration, lookup,
    and lifecycle operations in a thread-safe manner.

    Architectural Role
    -------------------
    ToolExecutor delegates execution to connectors via this manager.
    The manager itself contains no execution logic — it only routes
    requests and manages connector lifecycle.

    Design Properties
    -----------------
    • Thread-safe access using a re-entrant lock  
    • Prevents accidental connector overwrites  
    • Supports bulk registration  
    • Provides observability hooks (health)  
    • Supports graceful shutdown of external resources
    """

    def __init__(self) -> None:
        """Initialize an empty connector registry."""
        self._connectors: Dict[str, ExecutionConnector] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, connector: ExecutionConnector) -> None:
        """
        Register a connector under a unique name.

        Parameters
        ----------
        name : str
            Identifier used by tools to reference this connector.

        connector : ExecutionConnector
            Concrete connector implementation.

        Raises
        ------
        ValueError
            If name is invalid or already registered.

        TypeError
            If connector does not implement ExecutionConnector.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Connector must have a valid string name.")

        if not isinstance(connector, ExecutionConnector):
            raise TypeError("Connector must implement ExecutionConnector.")

        with self._lock:
            if name in self._connectors:
                raise ValueError(f"Connector '{name}' already registered.")
            self._connectors[name] = connector

    def register_many(self, connectors: Dict[str, ExecutionConnector]) -> None:
        """
        Atomically register multiple connectors.

        Raises
        ------
        ValueError
            If any connector name already exists.

        TypeError
            If any object does not implement ExecutionConnector.
        """
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
        """
        Retrieve a registered connector.

        Raises
        ------
        KeyError
            If the connector does not exist.
        """
        with self._lock:
            try:
                return self._connectors[name]
            except KeyError:
                raise KeyError(f"Connector '{name}' is not registered.") from None

    def has_connector(self, name: str) -> bool:
        """Return True if a connector with this name exists."""
        with self._lock:
            return name in self._connectors

    def list_connectors(self) -> Iterable[str]:
        """Return the names of all registered connectors."""
        with self._lock:
            return list(self._connectors.keys())

    # ------------------------------------------------------------------
    # Observability & Control
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, bool]:
        """
        Return the health status of all connectors.

        Used by stability monitoring to detect degraded backends.
        """
        with self._lock:
            return {name: conn.health() for name, conn in self._connectors.items()}

    def shutdown_all(self) -> None:
        """
        Gracefully shutdown all connectors.

        Intended for agent shutdown or restart scenarios.
        """
        with self._lock:
            for conn in self._connectors.values():
                try:
                    conn.shutdown()
                except Exception:
                    # Shutdown should not propagate connector errors
                    pass

    def __len__(self) -> int:
        """Return the number of registered connectors."""
        with self._lock:
            return len(self._connectors)
