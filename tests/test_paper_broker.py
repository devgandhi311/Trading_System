"""Deterministic tests for the authoritative persistent PAPER broker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backtesting.transaction_costs import TransactionCostModel
from broker.paper_broker import PaperBroker, PaperBrokerError
from models import OrderSide, OrderStatus, OrderType
from models.order_models import OrderRequest
from models.paper_models import MarketSnapshot
from storage import OperationalDatabase, OperationalRepository

NOW = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value


@pytest.fixture
def setup(tmp_path: Path):
    db = OperationalDatabase(tmp_path / "paper.db")
    db.initialize()
    repo = OperationalRepository(db)
    repo.create_trading_session("paper-session", started_at=NOW)
    clock = Clock()
    broker = PaperBroker(repo, session_id="paper-session", clock=clock,
                         transaction_cost_model=TransactionCostModel(slippage_percent=0.0))
    yield broker, repo, db, clock
    db.dispose()


def request(key="order-1", *, side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=10, limit=101.0, trigger=None, emergency=False):
    return OrderRequest(
        "NIFTY26000CE", side, quantity, order_type, NOW, limit_price=limit,
        trigger_price=trigger, metadata={"idempotency_key": key, "emergency_exit": emergency},
    )


def quote(quote_id="q1", *, bid=99.0, ask=100.0, quantity=10, timestamp=NOW):
    return MarketSnapshot(quote_id, timestamp, "NIFTY26000CE", "token-1", 99.5,
                          bid, ask, quantity)


def submit(broker, value, market_quote=None):
    return broker.submit_order(value, market_quote, instrument_token="token-1", lot_size=1)


def test_marketable_buy_limit(setup):
    broker, *_ = setup
    result = submit(broker, request(limit=101), quote(ask=100))
    assert result.status is OrderStatus.FILLED and result.average_fill_price == 100


def test_non_marketable_buy_limit(setup):
    broker, *_ = setup
    result = submit(broker, request(limit=99), quote(ask=100))
    assert result.status is OrderStatus.OPEN and not result.fills


def test_marketable_sell_limit(setup):
    broker, *_ = setup
    result = submit(broker, request(side=OrderSide.SELL, limit=99), quote(bid=100))
    assert result.status is OrderStatus.FILLED and result.average_fill_price == 100


def test_non_marketable_sell_limit(setup):
    broker, *_ = setup
    result = submit(broker, request(side=OrderSide.SELL, limit=101), quote(bid=100))
    assert result.status is OrderStatus.OPEN and not result.fills


def test_buy_sl_limit_dormant(setup):
    broker, *_ = setup
    result = submit(broker, request(order_type=OrderType.SL_LIMIT, limit=106, trigger=105),
                    quote(ask=104))
    assert result.lifecycle_state == "TRIGGER_PENDING" and not result.fills


def test_buy_sl_limit_triggered(setup):
    broker, *_ = setup
    result = submit(broker, request(order_type=OrderType.SL_LIMIT, limit=106, trigger=105),
                    quote(ask=105))
    assert result.status is OrderStatus.FILLED and result.average_fill_price == 105


def test_sell_sl_limit_dormant(setup):
    broker, *_ = setup
    result = submit(broker, request(side=OrderSide.SELL, order_type=OrderType.SL_LIMIT,
                                    limit=94, trigger=95), quote(bid=96, ask=97))
    assert result.lifecycle_state == "TRIGGER_PENDING"


def test_sell_sl_limit_triggered(setup):
    broker, *_ = setup
    result = submit(broker, request(side=OrderSide.SELL, order_type=OrderType.SL_LIMIT,
                                    limit=94, trigger=95), quote(bid=95, ask=96))
    assert result.status is OrderStatus.FILLED and result.average_fill_price == 95


def test_partial_fill_and_second_quote_complete(setup):
    broker, *_ = setup
    first = submit(broker, request(quantity=10), quote("q1", quantity=4))
    assert first.status is OrderStatus.PARTIALLY_FILLED and first.remaining_quantity == 6
    second = broker.process_quote("order-1", quote("q2", ask=101, quantity=6))
    assert second.status is OrderStatus.FILLED and [fill.quantity for fill in second.fills] == [4, 6]
    assert second.average_fill_price == pytest.approx(100.6)


def test_repeated_quote_does_not_duplicate_fill(setup):
    broker, *_ = setup
    supplied = quote("same", quantity=4)
    first = submit(broker, request(quantity=10), supplied)
    second = broker.process_quote("order-1", supplied)
    assert second.filled_quantity == first.filled_quantity == 4
    assert len(second.fills) == 1


def test_cancel_before_fill(setup):
    broker, repo, *_ = setup
    submit(broker, request(limit=90), quote())
    result = broker.cancel_order("order-1", reason="test")
    assert result.status is OrderStatus.CANCELLED and result.cancellation_reason == "test"
    assert repo.order_events("order-1")[-1].to_state == "CANCELLED"


def test_cancel_after_partial_fill_preserves_fill(setup):
    broker, *_ = setup
    submit(broker, request(quantity=10), quote(quantity=4))
    result = broker.cancel_order("order-1")
    assert result.status is OrderStatus.CANCELLED and result.filled_quantity == 4
    assert len(result.fills) == 1 and result.remaining_quantity == 6


def test_valid_modify_preserves_prior_events(setup):
    broker, repo, *_ = setup
    submit(broker, request(limit=90), quote())
    before = len(repo.order_events("order-1"))
    result = broker.modify_order("order-1", limit_price=101)
    assert result.requested_price == 101 and len(result.modification_history) == 1
    assert len(repo.order_events("order-1")) == before + 2


def test_invalid_modify_fails_closed(setup):
    broker, *_ = setup
    submit(broker, request(limit=90), quote())
    with pytest.raises(PaperBrokerError, match="positive"):
        broker.modify_order("order-1", limit_price=0)


def test_stale_quote_rejected(setup):
    broker, *_ = setup
    result = submit(broker, request(), quote(timestamp=NOW - timedelta(seconds=6)))
    assert result.status is OrderStatus.REJECTED and result.rejection_reason == "stale quote"


def test_crossed_quote_rejected(setup):
    broker, *_ = setup
    supplied = quote()
    object.__setattr__(supplied, "bid", 102.0)
    object.__setattr__(supplied, "ask", 101.0)
    result = submit(broker, request(), supplied)
    assert result.status is OrderStatus.REJECTED and result.rejection_reason == "crossed quote"


def test_zero_quantity_rejected(setup):
    broker, *_ = setup
    with pytest.raises(PaperBrokerError, match="quantity"):
        submit(broker, request(quantity=0))


def test_unavailable_liquidity_rejected(setup):
    broker, *_ = setup
    result = submit(broker, request(), quote(quantity=0))
    assert result.status is OrderStatus.REJECTED
    assert result.rejection_reason == "available liquidity is required"


def reopen(path, clock):
    db = OperationalDatabase(path)
    db.initialize()
    repo = OperationalRepository(db)
    return db, PaperBroker(repo, session_id="paper-session", clock=clock,
                           transaction_cost_model=TransactionCostModel(slippage_percent=0.0))


def test_restart_after_partial_fill(setup):
    broker, _, db, clock = setup
    submit(broker, request(quantity=10), quote(quantity=4))
    path = db.path
    db.dispose()
    restarted_db, restarted = reopen(path, clock)
    result = restarted.get_order("order-1")
    assert result.status is OrderStatus.PARTIALLY_FILLED and result.filled_quantity == 4
    result = restarted.process_quote("order-1", quote("q2", quantity=6))
    assert result.status is OrderStatus.FILLED and len(result.fills) == 2
    restarted_db.dispose()


def test_restart_after_full_fill_does_not_duplicate(setup):
    broker, _, db, clock = setup
    submit(broker, request(), quote("q1"))
    path = db.path
    db.dispose()
    restarted_db, restarted = reopen(path, clock)
    assert restarted.process_quote("order-1", quote("q1")).filled_quantity == 10
    assert len(restarted.get_order("order-1").fills) == 1
    restarted_db.dispose()


def test_emergency_market_exit(setup):
    _, repo, _, clock = setup
    broker = PaperBroker(repo, session_id="paper-session", clock=clock,
                         transaction_cost_model=TransactionCostModel(slippage_percent=0.0),
                         emergency_slippage_percent=1.0)
    result = submit(broker, request(order_type=OrderType.MARKET, limit=None, emergency=True),
                    quote(ask=100))
    assert result.status is OrderStatus.FILLED
    assert result.average_fill_price == pytest.approx(101.0)


def test_unauthorized_market_rejection(setup):
    broker, *_ = setup
    result = submit(broker, request(order_type=OrderType.MARKET, limit=None), quote())
    assert result.status is OrderStatus.REJECTED
    assert "authorized emergency" in result.rejection_reason


def test_transaction_cost_reconciliation(setup):
    broker, repo, *_ = setup
    result = submit(broker, request(quantity=10), quote(quantity=4, ask=100))
    result = broker.process_quote("order-1", quote("q2", quantity=6, ask=101))
    persisted = repo.fills_for_order("order-1")
    assert result.entry_turnover == pytest.approx(sum(fill.price * fill.quantity for fill in persisted))
    assert result.transaction_costs == pytest.approx(sum(fill.transaction_cost for fill in persisted))
    assert result.transaction_costs == pytest.approx(sum(broker.cost_model.estimate(
        price=fill.price, quantity=fill.quantity) for fill in persisted))


def test_order_timeout_expires(setup):
    broker, _, _, clock = setup
    submit(broker, request(limit=90))
    clock.value = NOW + timedelta(seconds=31)
    result = broker.process_quote("order-1", quote("late", timestamp=clock.value))
    assert result.status is OrderStatus.EXPIRED


def test_duplicate_submission_returns_authoritative_order(setup):
    broker, repo, *_ = setup
    first = submit(broker, request(), quote())
    second = submit(broker, request(), quote("different", bid=89, ask=90))
    assert second == first
    assert len(repo.fills_for_order("order-1")) == 1
