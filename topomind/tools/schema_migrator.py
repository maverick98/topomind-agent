from typing import Any, Dict


class SchemaMigrator:
    """
    Migrates tool outputs between schema versions.
    """

    def __init__(self):
        self._migrations = {}

    def register_migration(self, tool_name: str, from_v: str, to_v: str, func):
        self._migrations[(tool_name, from_v, to_v)] = func

    def migrate(self, tool_name: str, from_v: str, to_v: str, data: Dict[str, Any]):
        if from_v == to_v:
            return data

        key = (tool_name, from_v, to_v)
        if key not in self._migrations:
            raise ValueError(f"No migration path {key}")

        return self._migrations[key](data)
