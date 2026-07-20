"""Tests for authoritative transactional operational persistence."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from storage import (
    ForbiddenSecretMetadataError,
    InvalidOrderTransitionError,
    OperationalDatabase,
    OperationalRepository,
    OptimisticLockError,
    UnsupportedSchemaVersionError,
)
from storage.repositories import validate_metadata
from storage.models import OrderIntentRecord, SchemaVersion, TradingSession


@pytest.fixture
def database(tmp_path: Path):
    db = OperationalDatabase(tmp_path / "state.db")
    db.initialize()
    yield db
    db.dispose()


@pytest.fixture
def repository(database):
    repo = OperationalRepository(database)
    repo.create_trading_session("session-1")
    return repo


def reserve(repo, key="order-1", **overrides):
    values = dict(
        idempotency_key=key, client_order_id=f"client-{key}", session_id="session-1",
        instrument_token="26000", symbol="NIFTY26000CE", side="BUY", order_type="LIMIT",
        quantity=75, limit_price=100.0, product_type="MIS", validity="DAY", local_state="PENDING",
    )
    values.update(overrides)
    return repo.reserve_order_intent(**values)


def reopen(path: Path):
    db = OperationalDatabase(path)
    db.initialize()
    return db, OperationalRepository(db)


def test_database_initialization_enables_sqlite_safety(database) -> None:
    with database.transaction() as session:
        assert session.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert session.execute(text("PRAGMA busy_timeout")).scalar_one() > 0
        assert session.execute(text("PRAGMA journal_mode")).scalar_one().lower() in {"wal", "memory"}
        tables = {row[0] for row in session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
    assert {"trading_sessions", "order_intents", "fills", "positions", "portfolio_snapshots"} <= tables


def test_schema_version_creation(database) -> None:
    with database.transaction() as session:
        assert session.scalar(select(SchemaVersion.version)) == 2


def test_unsupported_newer_schema_rejected(tmp_path) -> None:
    path = tmp_path / "newer.db"
    db = OperationalDatabase(path)
    db.initialize()
    with db.transaction() as session:
        session.execute(text("UPDATE schema_versions SET version=999"))
    db.dispose()
    newer = OperationalDatabase(path)
    with pytest.raises(UnsupportedSchemaVersionError, match="newer"):
        newer.initialize()
    newer.dispose()


def test_transaction_commit(database) -> None:
    with database.transaction() as session:
        session.add(TradingSession(id="committed", mode="PAPER", trading_date="2026-07-15",
                                   status="OPEN", started_at=__import__("utils.time_utils", fromlist=["now_ist"]).now_ist()))
    with database.transaction() as session:
        assert session.get(TradingSession, "committed") is not None


def test_transaction_rollback(database) -> None:
    with pytest.raises(RuntimeError):
        with database.transaction() as session:
            session.add(TradingSession(id="rolled-back", mode="PAPER", trading_date="2026-07-15",
                                       status="OPEN", started_at=__import__("utils.time_utils", fromlist=["now_ist"]).now_ist()))
            raise RuntimeError("force rollback")
    with database.transaction() as session:
        assert session.get(TradingSession, "rolled-back") is None


def test_foreign_keys_are_enforced(database) -> None:
    repo = OperationalRepository(database)
    with pytest.raises(IntegrityError):
        repo.reserve_order_intent(idempotency_key="fk", client_order_id="fk", session_id="missing",
                                  instrument_token="1", symbol="NIFTY", side="BUY", order_type="LIMIT",
                                  quantity=1, product_type="MIS", validity="DAY", local_state="PENDING")


def test_duplicate_idempotency_reservation_returns_authoritative_record(repository) -> None:
    first = reserve(repository)
    second = reserve(repository, client_order_id="different-client")
    assert first.id == second.id
    assert len(repository.order_events("order-1")) == 1


def test_concurrent_duplicate_reservation_creates_one_record(repository, database) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: reserve(repository, "concurrent"), range(8)))
    assert len({row.id for row in rows}) == 1
    with database.transaction() as session:
        count = session.scalar(select(func.count()).select_from(OrderIntentRecord).where(
            OrderIntentRecord.idempotency_key == "concurrent"))
    assert count == 1


def test_invalid_state_transition_fails(repository) -> None:
    row = reserve(repository)
    with pytest.raises(InvalidOrderTransitionError):
        repository.transition_order("order-1", "FILLED", expected_version=row.version)


def test_transition_creates_event(repository) -> None:
    row = reserve(repository)
    changed = repository.transition_order("order-1", "IN_FLIGHT", expected_version=row.version)
    events = repository.order_events("order-1")
    assert changed.local_state == "IN_FLIGHT"
    assert [(event.from_state, event.to_state) for event in events] == [(None, "PENDING"), ("PENDING", "IN_FLIGHT")]


def test_optimistic_lock_conflict(repository) -> None:
    row = reserve(repository)
    repository.transition_order("order-1", "IN_FLIGHT", expected_version=row.version)
    with pytest.raises(OptimisticLockError):
        repository.transition_order("order-1", "IN_FLIGHT", expected_version=row.version)


def test_unknown_survives_restart(repository, database) -> None:
    path = database.path
    row = reserve(repository)
    row = repository.transition_order("order-1", "IN_FLIGHT", expected_version=row.version)
    repository.transition_order("order-1", "UNKNOWN", expected_version=row.version,
                                failure_classification="TIMEOUT")
    database.dispose()
    restarted, repo = reopen(path)
    recovered = repo.get_order_intent("order-1")
    assert recovered.local_state == "UNKNOWN" and recovered.reconciliation_required is True
    restarted.dispose()


def test_reconciliation_required_survives_restart(repository, database) -> None:
    path = database.path
    row = reserve(repository, "reconciliation-required")
    row = repository.transition_order(
        "reconciliation-required", "IN_FLIGHT", expected_version=row.version
    )
    row = repository.transition_order(
        "reconciliation-required", "UNKNOWN", expected_version=row.version,
        failure_classification="TIMEOUT",
    )
    repository.transition_order(
        "reconciliation-required", "RECONCILIATION_REQUIRED", expected_version=row.version
    )
    database.dispose()
    restarted, repo = reopen(path)
    recovered = repo.get_order_intent("reconciliation-required")
    assert recovered.local_state == "RECONCILIATION_REQUIRED"
    assert recovered.reconciliation_required is True
    restarted.dispose()


def test_partial_fill_survives_restart(repository, database) -> None:
    path = database.path
    row = reserve(repository)
    row = repository.transition_order("order-1", "IN_FLIGHT", expected_version=row.version)
    repository.transition_order("order-1", "PARTIALLY_FILLED", expected_version=row.version,
                                filled_quantity=25)
    repository.add_fill("fill-1", idempotency_key="order-1", quantity=25, price=99.5)
    database.dispose()
    restarted, repo = reopen(path)
    assert repo.get_order_intent("order-1").filled_quantity == 25
    assert [(fill.quantity, fill.price) for fill in repo.fills_for_order("order-1")] == [(25, 99.5)]
    restarted.dispose()


def test_open_position_survives_restart(repository, database) -> None:
    path = database.path
    repository.save_position("position-1", session_id="session-1", symbol="NIFTY", quantity=75,
                             average_price=100.0)
    database.dispose()
    restarted, repo = reopen(path)
    assert [(row.id, row.quantity) for row in repo.open_positions()] == [("position-1", 75)]
    restarted.dispose()


def test_current_equity_survives_restart(repository, database) -> None:
    path = database.path
    repository.save_portfolio_snapshot(session_id="session-1", initial_capital=1_000_000,
                                       current_equity=987_500, realized_pnl=-10_000,
                                       unrealized_pnl=0, accrued_costs=2_500)
    database.dispose()
    restarted, repo = reopen(path)
    assert repo.latest_portfolio().current_equity == 987_500
    restarted.dispose()


def test_kill_switch_survives_restart(repository, database) -> None:
    path = database.path
    repository.trigger_kill_switch(reason="reconciliation mismatch", source="test", session_id="session-1")
    database.dispose()
    restarted, repo = reopen(path)
    state = repo.current_kill_switch()
    assert state.active is True and state.reason == "reconciliation mismatch"
    restarted.dispose()


@pytest.mark.parametrize("key", [
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "passwd",
    "mpin",
    "pin",
    "otp",
    "totp",
    "totp_secret",
    "session_token",
    "bearer",
    "client_secret",
])
@pytest.mark.parametrize("nested", [False, True])
def test_forbidden_secret_metadata_rejected(repository, key, nested) -> None:
    metadata = {"safe": [{"nested": {key: "secret-value"}}]} if nested else {key: "secret-value"}
    with pytest.raises(ForbiddenSecretMetadataError):
        reserve(repository, f"secret-{key}-{nested}", metadata=metadata)
    assert repository.get_order_intent(f"secret-{key}-{nested}") is None


@pytest.mark.parametrize("key", ["API_KEY", "Authorization", "BEARER"])
def test_forbidden_secret_metadata_rejected_mixed_case(repository, key) -> None:
    metadata = {key: "secret-value"}
    with pytest.raises(ForbiddenSecretMetadataError):
        reserve(repository, f"secret-mixed-{key}", metadata=metadata)


def test_forbidden_secret_in_nested_mapping_rejected(repository) -> None:
    metadata = {"outer": {"inner": {"api_key": "secret"}}}
    with pytest.raises(ForbiddenSecretMetadataError):
        reserve(repository, "secret-nested-mapping", metadata=metadata)


def test_forbidden_secret_in_list_rejected(repository) -> None:
    metadata = [{"safe": "value"}, {"api_key": "secret"}]
    with pytest.raises(ForbiddenSecretMetadataError):
        reserve(repository, "secret-list", metadata=metadata)


def test_forbidden_secret_in_tuple_rejected(repository) -> None:
    metadata = ({"safe": "value"}, {"api_key": "secret"})
    with pytest.raises(ForbiddenSecretMetadataError):
        reserve(repository, "secret-tuple", metadata=metadata)


def test_forbidden_secret_in_set_rejected(repository) -> None:
    metadata = {"safe", "api_key"}
    with pytest.raises(ForbiddenSecretMetadataError):
        reserve(repository, "secret-set", metadata=metadata)


def test_transaction_rollback_on_forbidden_metadata(database) -> None:
    repo = OperationalRepository(database)
    repo.create_trading_session(session_id="session-rollback", mode="PAPER")
    initial_count = len(repo.session_orders("session-rollback"))
    with pytest.raises(ForbiddenSecretMetadataError):
        repo.reserve_order_intent(
            idempotency_key="rollback",
            client_order_id="client-rollback",
            session_id="session-rollback",
            instrument_token="26000",
            symbol="NIFTY26000CE",
            side="BUY",
            order_type="LIMIT",
            quantity=75,
            limit_price=100.0,
            product_type="MIS",
            validity="DAY",
            local_state="PENDING",
            metadata={"api_key": "secret"},
        )
    assert len(repo.session_orders("session-rollback")) == initial_count


def test_safe_metadata_is_accepted(repository) -> None:
    record = reserve(repository, "safe-meta", metadata={"strategy": "credit_spread", "confidence": 0.8})
    assert record.idempotency_key == "safe-meta"
    assert repository.get_order_intent("safe-meta").metadata_json == {"strategy": "credit_spread", "confidence": 0.8}


@pytest.mark.parametrize("key", ["token_count", "model_tokenizer", "authorization_status"])
def test_harmless_token_related_keys_accepted(repository, key) -> None:
    record = reserve(repository, f"harmless-{key}", metadata={key: "safe-value"})
    assert record.idempotency_key == f"harmless-{key}"


def test_forbidden_secret_value_not_in_exception_message() -> None:
    try:
        validate_metadata({"api_key": "TOP-SECRET"})
    except ForbiddenSecretMetadataError as exc:
        assert "TOP-SECRET" not in str(exc)
        assert "api_key" in str(exc)
