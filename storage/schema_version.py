"""Operational database schema version contract."""

CURRENT_SCHEMA_VERSION = 2


class UnsupportedSchemaVersionError(RuntimeError):
    """Raised when a database was created by newer unsupported code."""