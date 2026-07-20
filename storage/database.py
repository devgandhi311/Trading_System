"""SQLite engine and explicit transactional session lifecycle."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from storage.models import Base, SchemaVersion
from storage.schema_version import CURRENT_SCHEMA_VERSION, UnsupportedSchemaVersionError

DEFAULT_DATABASE_PATH = Path("data/live/trading_state.db")


class OperationalDatabase:
    """Own the authoritative SQLite engine and scoped transactions."""

    def __init__(self, path: str | Path = DEFAULT_DATABASE_PATH, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.path.resolve().as_posix()}",
            future=True,
            connect_args={"check_same_thread": False, "timeout": busy_timeout_ms / 1000},
        )
        self._configure_sqlite(self.engine, busy_timeout_ms)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    @staticmethod
    def _configure_sqlite(engine: Engine, busy_timeout_ms: int) -> None:
        @event.listens_for(engine, "connect")
        def configure(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
            finally:
                cursor.close()

    def initialize(self) -> None:
        """Create tables and validate the single supported schema version."""

        Base.metadata.create_all(self.engine)
        with self.transaction() as session:
            row = session.scalar(select(SchemaVersion).order_by(SchemaVersion.id.desc()).limit(1))
            if row is None:
                session.add(SchemaVersion(version=CURRENT_SCHEMA_VERSION))
            elif row.version > CURRENT_SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"Database schema {row.version} is newer than supported {CURRENT_SCHEMA_VERSION}"
                )
            elif row.version < CURRENT_SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"Database schema {row.version} requires a migration to {CURRENT_SCHEMA_VERSION}"
                )

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """Commit on success and roll back every failure."""

        session = self.session_factory()
        try:
            with session.begin():
                yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Release pooled SQLite connections."""

        self.engine.dispose()

