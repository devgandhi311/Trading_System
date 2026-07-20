"""Deterministic tests for the persistent PAPER reconciliation service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from models import OrderSide, OrderStatus, OrderType
from models.order_models import OrderIntentRecord, OrderIntentStatus
from models.paper_models import PaperOrder, PaperPosition, ReconciliationOutcome, ReconciliationResult
from runtime.reconciliation import PaperReconciliationService
from storage import OperationalDatabase, OperationalRepository
from utils.exceptions import BrokerError
from utils.time_utils import now_ist

NOW = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def database(tmp_path: Path):
    db = OperationalDatabase(tmp_path / "recon.db")
    db.initialize()
    yield db
    db.dispose()


@pytest.fixture
def repository(database):
    repo = OperationalRepository(database)
    repo.create_trading_session("recon-session", started_at=NOW)
    return repo


def local_order(repo, key="order-1", **overrides):
    values = dict(
        idempotency_key=key, client_order_id=f"client-{key}", session_id="recon-session",
        instrument_token="26000", symbol="NIFTY26000CE", side="BUY", order_type="LIMIT",
        quantity=10, limit_price=100.0, product_type="MIS", validity="DAY", local_state="OPEN",
        filled_quantity=0,
    )
    values.update(overrides)
    return repo.reserve_order_intent(**values)


def local_fill(repo, order_key="order-1", fill_id="fill-1", quantity=5, price=100.0, cost=1.0):
    intent = repo.get_order_intent(order_key)
    return repo.add_fill(fill_id, idempotency_key=order_key, quantity=quantity, price=price,
                         transaction_cost=cost, filled_at=NOW)


def local_position(repo, position_id="pos-1", symbol="NIFTY", quantity=10, average_price=100.0):
    return repo.save_position(position_id, session_id="recon-session", symbol=symbol, quantity=quantity,
                              average_price=average_price)


def broker_order(order_id="order-1", **overrides):
    defaults = dict(
        order_id=order_id, created_at=NOW, updated_at=NOW, symbol="NIFTY26000CE",
        instrument_token="26000", strategy_id="strategy-1", leg_id="leg-1", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, quantity=10, lot_size=1, status=OrderStatus.OPEN,
        filled_quantity=0, remaining_quantity=10, transaction_costs=0.0,
    )
    defaults.update(overrides)
    if "filled_quantity" in overrides:
        defaults["remaining_quantity"] = max(0, defaults["quantity"] - overrides["filled_quantity"])
    elif "quantity" in overrides:
        defaults["remaining_quantity"] = overrides["quantity"]
    return PaperOrder(**defaults)


def broker_position(position_id="pos-1", **overrides):
    from models.strategy_models import StrategyLeg
    defaults = dict(
        position_id=position_id, opened_at=NOW, updated_at=NOW, symbol="NIFTY",
        instrument_token="26000", strategy_id="strategy-1", strategy_type="iron_condor",
        legs=(StrategyLeg(symbol="NIFTY26000CE", side=OrderSide.BUY, quantity=10),),
        side=OrderSide.BUY, quantity=10, lot_size=1, status=OrderStatus.OPEN,
        entry_price=100.0, current_price=100.0, risk_amount=1000.0, entry_reason="test",
        realized_pnl=0.0, transaction_costs=0.0,
    )
    defaults.update(overrides)
    return PaperPosition(**defaults)


def test_clean_state(repository):
    local_order(repository, "order-1")
    broker_orders = {"order-1": broker_order("order-1")}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.CLEAN
    assert result.entry_blocked is False
    assert result.exits_allowed is True
    assert result.affected_order_ids == ()
    assert result.matched_orders == 1


def test_local_only_order(repository):
    local_order(repository, "order-1")
    broker_orders = {}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.LOCAL_ORDER_MISSING
    assert result.entry_blocked is True
    assert result.exits_allowed is True
    assert "order-1" in result.affected_order_ids


def test_broker_only_order(repository):
    broker_orders = {"order-1": broker_order("order-1")}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.BROKER_ORDER_MISSING
    assert result.entry_blocked is True
    assert result.exits_allowed is True
    assert "order-1" in result.affected_order_ids


def test_state_mismatch(repository):
    local_order(repository, "order-1", local_state="OPEN")
    broker_orders = {"order-1": broker_order("order-1", status=OrderStatus.FILLED, lifecycle_state="FILLED")}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.ORDER_STATE_MISMATCH
    assert result.entry_blocked is True
    assert "order-1" in result.affected_order_ids


def test_fill_mismatch(repository):
    local_order(repository, "order-1", local_state="PARTIALLY_FILLED", filled_quantity=5)
    broker_orders = {"order-1": broker_order("order-1", filled_quantity=10, lifecycle_state="PARTIALLY_FILLED")}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.FILL_MISMATCH
    assert result.entry_blocked is True
    assert "order-1" in result.affected_order_ids


def test_quantity_mismatch(repository):
    local_order(repository, "order-1", quantity=10)
    broker_orders = {"order-1": broker_order("order-1", quantity=20)}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.QUANTITY_MISMATCH
    assert result.entry_blocked is True
    assert "order-1" in result.affected_order_ids


def test_price_mismatch(repository):
    local_order(repository, "order-1", local_state="FILLED", filled_quantity=10, metadata={"average_fill_price": 100.0, "transaction_costs": 1.0})
    broker_orders = {"order-1": broker_order("order-1", average_fill_price=101.0, transaction_costs=1.0, lifecycle_state="FILLED", filled_quantity=10)}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.PRICE_MISMATCH
    assert result.entry_blocked is True
    assert "order-1" in result.affected_order_ids


def test_transaction_cost_mismatch(repository):
    local_order(repository, "order-1", local_state="FILLED", filled_quantity=10)
    broker_orders = {"order-1": broker_order("order-1", transaction_costs=5.0, lifecycle_state="FILLED", filled_quantity=10)}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.COST_MISMATCH
    assert result.entry_blocked is True
    assert "order-1" in result.affected_order_ids


def test_position_mismatch(repository):
    local_position(repository, "pos-1", quantity=10, average_price=100.0)
    broker_positions = {"pos-1": broker_position("pos-1", quantity=20, entry_price=100.0)}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: {}, lambda: {}, lambda: broker_positions)
    assert result.outcome == ReconciliationOutcome.POSITION_MISMATCH
    assert result.entry_blocked is True
    assert "pos-1" in result.affected_position_ids


def test_unknown_order(repository):
    local_order(repository, "order-1", local_state="UNKNOWN")
    broker_orders = {"order-1": broker_order("order-1", lifecycle_state="UNKNOWN")}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.UNKNOWN_ORDER
    assert result.entry_blocked is True
    assert result.exits_allowed is True
    assert "order-1" in result.affected_order_ids


def test_ambiguous_state(repository):
    local_order(repository, "order-1", local_state="OPEN")
    broker_orders = {"order-2": broker_order("order-2", lifecycle_state="OPEN")}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.AMBIGUOUS
    assert result.entry_blocked is True
    assert result.exits_allowed is True
    assert "order-1" in result.affected_order_ids
    assert "order-2" in result.affected_order_ids


def test_broker_unavailable(repository):
    local_order(repository, "order-1")
    service = PaperReconciliationService(repository, "recon-session")

    def fail_provider():
        raise BrokerError("broker unavailable")

    result = service.reconcile(fail_provider, fail_provider, fail_provider)
    assert result.outcome == ReconciliationOutcome.BROKER_UNAVAILABLE
    assert result.entry_blocked is True
    assert result.exits_allowed is True


def test_eventual_consistency_resolves_cleanly(repository):
    local_order(repository, "order-1", local_state="SUBMITTED")
    broker_orders = {"order-1": broker_order("order-1", lifecycle_state="SUBMITTED")}
    service = PaperReconciliationService(repository, "recon-session", eventual_consistency_seconds=120.0)
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {}, now=NOW)
    assert result.outcome == ReconciliationOutcome.EVENTUAL_CONSISTENCY_PENDING
    assert result.entry_blocked is False
    assert result.valid_until is not None
    assert (result.valid_until - NOW).total_seconds() == pytest.approx(120.0, abs=1.0)


def test_eventual_consistency_timeout(repository):
    past = NOW - timedelta(seconds=200)
    local_order(repository, "order-1", local_state="UNKNOWN")
    broker_orders = {"order-1": broker_order("order-1", lifecycle_state="UNKNOWN")}
    service = PaperReconciliationService(repository, "recon-session", eventual_consistency_seconds=120.0)
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {}, now=past)
    assert result.outcome == ReconciliationOutcome.UNKNOWN_ORDER
    assert result.valid_until is not None


def test_stale_clean_reconciliation(repository):
    service = PaperReconciliationService(repository, "recon-session", clean_expiry_seconds=60.0)
    old = NOW - timedelta(seconds=120)
    result = service.reconcile(lambda: {}, lambda: {}, lambda: {}, now=old)
    assert result.outcome == ReconciliationOutcome.CLEAN
    assert repository.is_reconciliation_stale("recon-session", clean_expiry_seconds=60.0) is True


def test_restart_recovery(database):
    repo = OperationalRepository(database)
    repo.create_trading_session("recon-session", started_at=NOW)
    local_order(repo, "order-1", local_state="OPEN")
    broker_orders = {"order-1": broker_order("order-1", lifecycle_state="FILLED")}
    service = PaperReconciliationService(repo, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.ORDER_STATE_MISMATCH
    database.dispose()
    restarted = OperationalDatabase(database.path)
    restarted.initialize()
    restarted_repo = OperationalRepository(restarted)
    recovered_service = PaperReconciliationService(restarted_repo, "recon-session")
    recovered = recovered_service.latest_reconciliation()
    assert recovered is not None
    assert recovered.outcome == ReconciliationOutcome.ORDER_STATE_MISMATCH
    assert recovered.entry_blocked is True
    assert recovered_service.has_unresolved_blocking_reconciliation() is True
    restarted.dispose()


def test_entry_blocked_but_exit_allowed(repository):
    local_order(repository, "order-1", local_state="OPEN")
    broker_orders = {"order-1": broker_order("order-1", lifecycle_state="FILLED")}
    local_position(repository, "pos-1")
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.ORDER_STATE_MISMATCH
    assert result.entry_blocked is True
    assert result.exits_allowed is True
    assert "order-1" in result.affected_order_ids


def test_reconciliation_result_persistence(repository):
    local_order(repository, "order-1", local_state="OPEN")
    broker_orders = {"order-1": broker_order("order-1", lifecycle_state="FILLED")}
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    history = service.reconciliation_history()
    assert len(history) == 1
    assert history[0].outcome == ReconciliationOutcome.ORDER_STATE_MISMATCH
    assert history[0].reason != ""
    assert history[0].session_id == "recon-session"


def test_multiple_mismatches_accumulate(repository):
    local_order(repository, "order-1", local_state="OPEN")
    local_order(repository, "order-2", local_state="OPEN")
    broker_orders = {
        "order-1": broker_order("order-1", lifecycle_state="FILLED"),
        "order-2": broker_order("order-2", lifecycle_state="FILLED"),
    }
    service = PaperReconciliationService(repository, "recon-session")
    result = service.reconcile(lambda: broker_orders, lambda: {}, lambda: {})
    assert result.outcome == ReconciliationOutcome.ORDER_STATE_MISMATCH
    assert len(result.affected_order_ids) == 2
    assert result.entry_blocked is True
