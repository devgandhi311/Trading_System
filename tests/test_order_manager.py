from datetime import datetime, timezone

import pytest

from broker.order_manager import OrderManager
from models import Mode, OrderSide, OrderStatus, OrderType
from models.order_models import OrderRequest, TradeState
from utils.exceptions import BrokerError, OrderValidationError


def _limit_order() -> OrderRequest:
    return OrderRequest(
        symbol="NIFTY",
        side=OrderSide.BUY,
        quantity=75,
        order_type=OrderType.LIMIT,
        timestamp=datetime.now(timezone.utc),
        limit_price=100.0,
    )


def test_paper_place_modify_cancel_order() -> None:
    manager = OrderManager(mode=Mode.PAPER)

    placed = manager.place_order(_limit_order())
    assert placed.status == OrderStatus.OPEN
    assert placed.raw_response["simulated"] is True

    modified = manager.modify_order(placed.order_id, 101.0)
    assert modified.status == OrderStatus.OPEN
    assert modified.raw_response["new_price"] == 101.0

    cancelled = manager.cancel_order(placed.order_id)
    assert cancelled.status == OrderStatus.CANCELLED


def test_backtest_simulates_orders() -> None:
    manager = OrderManager(mode=Mode.BACKTEST)

    response = manager.place_order(_limit_order())

    assert response.order_id.startswith("backtest-")
    assert response.status == OrderStatus.OPEN


def test_market_order_rejected_unless_emergency_exit() -> None:
    manager = OrderManager(mode=Mode.PAPER)
    request = OrderRequest(
        symbol="NIFTY",
        side=OrderSide.SELL,
        quantity=75,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(timezone.utc),
    )

    with pytest.raises(OrderValidationError, match="emergency exits"):
        manager.place_order(request)

    emergency = OrderRequest(
        symbol="NIFTY",
        side=OrderSide.SELL,
        quantity=75,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(timezone.utc),
        metadata={"emergency_exit": True},
    )
    response = manager.place_order(emergency)
    assert response.status == OrderStatus.FILLED


def test_live_orders_blocked_without_live_enabled_config() -> None:
    manager = OrderManager(mode=Mode.LIVE)

    with pytest.raises(BrokerError, match="mode=LIVE"):
        manager.place_order(_limit_order())


def test_squareoff_uses_limit_order() -> None:
    manager = OrderManager(mode=Mode.PAPER)
    position = TradeState(
        trade_id="trade-1",
        symbol="NIFTY",
        side=OrderSide.BUY,
        quantity=75,
        entry_time=datetime.now(timezone.utc),
        entry_price=100.0,
        status=OrderStatus.OPEN,
    )

    response = manager.squareoff_position(position)

    assert response.side == OrderSide.SELL
    assert response.status == OrderStatus.OPEN
    assert manager.orders[response.order_id].raw_response["simulated"] is True


def test_cancel_all_orders_cancels_open_simulated_orders() -> None:
    manager = OrderManager(mode=Mode.PAPER)
    first = manager.place_order(_limit_order())
    second = manager.place_order(_limit_order())

    responses = manager.cancel_all_orders()

    assert {response.order_id for response in responses} == {first.order_id, second.order_id}
    assert all(response.status == OrderStatus.CANCELLED for response in responses)
