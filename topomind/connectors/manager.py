from __future__ import annotations

from typing import Dict, Iterable
from threading import RLock
from pathlib import Path
import json
import logging

from .base import ExecutionConnector
from .rest_connector import RestConnector

logger = logging.getLogger(__name__)


class ConnectorManager:

    def __init__(self, storage_path: str = "connectors.json") -> None:
        self._connectors: Dict[str, ExecutionConnector] = {}
        self._status: Dict[str, str] = {}       # "active" | "inactive"
        self._metadata: Dict[str, dict] = {}    # persisted connector config
        self._lock = RLock()
        self._storage = Path(storage_path)

        self._load_from_disk()

    # ==========================================================
    # Strict Registration (Backward Compatible)
    # ==========================================================

    def register(self, name: str, connector: ExecutionConnector) -> None:
        self._validate(name, connector)

        with self._lock:
            if name in self._connectors:
                raise ValueError(f"Connector '{name}' already registered.")

            self._connectors[name] = connector
            self._status[name] = "active"
            self._metadata[name] = {}
            #self._save_to_disk()

    # ==========================================================
    # Idempotent Registration (Persistent)
    # ==========================================================

    def register_or_update(
        self,
        name: str,
        connector: ExecutionConnector,
        metadata: dict | None = None,
    ) -> str:

        self._validate(name, connector)

        with self._lock:
            if name in self._connectors:
                self._connectors[name] = connector
                self._status[name] = "active"
                if metadata:
                    self._metadata[name] = metadata
                self._save_to_disk()
                logger.info(f"[CONNECTOR] Updated '{name}'")
                return "updated"

            self._connectors[name] = connector
            self._status[name] = "active"
            self._metadata[name] = metadata or {}
            self._save_to_disk()
            logger.info(f"[CONNECTOR] Registered '{name}'")
            return "registered"

    # ==========================================================
    # Lookup
    # ==========================================================

    def get(self, name: str) -> ExecutionConnector:
        with self._lock:
            if name not in self._connectors:
                raise KeyError(f"Connector '{name}' is not registered.")

            if self._status.get(name) != "active":
                raise RuntimeError(f"Connector '{name}' is inactive.")

            return self._connectors[name]

    def is_registered(self, name: str) -> bool:
        with self._lock:
            return name in self._connectors

    def is_active(self, name: str) -> bool:
        with self._lock:
            return self._status.get(name) == "active"

    def list_connectors(self) -> Iterable[Dict[str, str]]:
        with self._lock:
            return [
                {
                    "name": name,
                    "status": self._status.get(name, "active"),
                    "type": self._metadata.get(name, {}).get("type", "unknown"),
                }
                for name in self._connectors
            ]

    # ==========================================================
    # Lifecycle
    # ==========================================================

    def deploy(self, name: str) -> None:
        with self._lock:
            if name not in self._connectors:
                raise KeyError(f"Connector '{name}' not found.")

            self._status[name] = "active"
            self._save_to_disk()
            logger.info(f"[CONNECTOR] Deployed '{name}'")

    def undeploy(self, name: str) -> None:
        with self._lock:
            if name not in self._connectors:
                raise KeyError(f"Connector '{name}' not found.")

            self._status[name] = "inactive"
            self._save_to_disk()
            logger.info(f"[CONNECTOR] Undeployed '{name}'")

    # ==========================================================
    # Observability
    # ==========================================================

    def health(self) -> Dict[str, bool]:
        status = {}

        with self._lock:
            for name, conn in self._connectors.items():
                try:
                    if self._status.get(name) != "active":
                        status[name] = False
                    else:
                        status[name] = conn.health()
                except Exception:
                    status[name] = False

        return status

    def shutdown_all(self) -> None:
        with self._lock:
            for name, conn in self._connectors.items():
                try:
                    conn.shutdown()
                except Exception:
                    logger.warning(
                        f"Shutdown failed for connector '{name}'"
                    )

    def __len__(self) -> int:
        with self._lock:
            return len(self._connectors)

    # ==========================================================
    # Persistence
    # ==========================================================

    def _save_to_disk(self) -> None:
        data = {}

        for name in self._connectors:
            data[name] = {
                "metadata": self._metadata.get(name, {}),
                "status": self._status.get(name, "active"),
            }

        try:
            with self._storage.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[CONNECTOR] Failed to persist: {e}")

    def _load_from_disk(self) -> None:
        if not self._storage.exists():
            return

        try:
            with self._storage.open() as f:
                data = json.load(f)

            for name, entry in data.items():
                metadata = entry.get("metadata", {})
                status = entry.get("status", "active")

                connector = self._reconstruct_connector(metadata)
                if connector:
                    self._connectors[name] = connector
                    self._status[name] = status
                    self._metadata[name] = metadata

            logger.info("[CONNECTOR] Loaded persisted connectors")

        except Exception as e:
            logger.warning(f"[CONNECTOR] Failed to load from disk: {e}")

    def _reconstruct_connector(self, metadata: dict) -> ExecutionConnector | None:
        connector_type = metadata.get("type")

        if connector_type == "rest":
            return RestConnector(
                base_url=metadata.get("base_url"),
                method=metadata.get("method", "POST"),
                timeout_seconds=metadata.get("timeout_seconds", 10),
            )

        return None

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate(self, name: str, connector: ExecutionConnector) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Connector must have a valid string name.")

        if not isinstance(connector, ExecutionConnector):
            raise TypeError(
                "Connector must implement ExecutionConnector."
            )
