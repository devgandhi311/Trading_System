"""Authoritative operational persistence API."""

from storage.database import DEFAULT_DATABASE_PATH, OperationalDatabase
from storage.repositories import (
    ForbiddenSecretMetadataError,
    InvalidOrderTransitionError,
    OperationalRepository,
    OptimisticLockError,
    validate_metadata,
)
from storage.schema_version import CURRENT_SCHEMA_VERSION, UnsupportedSchemaVersionError

__all__ = [
    "CURRENT_SCHEMA_VERSION", "DEFAULT_DATABASE_PATH", "ForbiddenSecretMetadataError",
    "InvalidOrderTransitionError", "OperationalDatabase", "OperationalRepository",
    "OptimisticLockError", "UnsupportedSchemaVersionError", "validate_metadata",
]
