"""Tests for execution pricing and order validation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from execution import (
    OrderValidator,
    build_profit_booking_order,
    build_short_option_stop_loss_order,
    calculate_limit_price,
    calculate_sl_limit_price,
    calculate_trigger_price,
    validate_order_price,
)
from models import OrderSide, OrderType
from models.order_models import OrderRequest
from utils.exceptions import OrderValidationError
from utils.time_utils import KOLKATA_TZ


def test_calculate_limit_price_uses_best_bid_ask_with_buffer() -> None:
    assert calculate_limit_price(OrderSide.BUY, ltp=100.0, bid=99.8, ask=100.1, buffer=0.1) == pytest.approx(100.2)
    assert calculate_limit_price(OrderSide.SELL, ltp=100.0, bid=99.8, ask=100.1, buffer=0.1) == pytest.approx(99.7)


def test_calculate_short_option_stop_loss_prices() -> None:
    trigger = calculate_trigger_price(entry_price=100.0, stop_loss_percent=25.0, side=OrderSide.SELL)
    limit = calculate_sl_limit_price(trigger, OrderSide.BUY, buffer=2.0)

    assert trigger == pytest.approx(125.0)
    assert limit == pytest.approx(127.0)


def test_profit_booking_order_is_limit_buy_and_validates() -> None:
    order = build_profit_booking_order(
        symbol="NIFTY26000CE",
        quantity=75,
        quote={"ltp": 100.0, "bid": 99.95, "ask": 100.05},
        buffer=0.01,
    )

    result = validate_order_price(order, {"ltp": 100.0, "bid": 99.95, "ask": 100.05})

    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.LIMIT
    assert order.metadata["short_option_exit"] is True
    assert result.approved is True


def test_short_option_stop_loss_order_is_buy_sl_limit() -> None:
    order = build_short_option_stop_loss_order(
        symbol="NIFTY26000CE",
        quantity=75,
        entry_price=100.0,
        stop_loss_percent=25.0,
        buffer=1.0,
    )

    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.SL_LIMIT
    assert order.trigger_price == pytest.approx(125.0)
    assert order.limit_price == pytest.approx(126.0)
    assert validate_order_price(order, {"ltp": 125.9, "bid": 125.8, "ask": 126.0}).approved is True


def test_option_selling_entry_rejects_market_order() -> None:
    order = OrderRequest(
        symbol="NIFTY26000CE",
        side=OrderSide.SELL,
        quantity=75,
        order_type=OrderType.MARKET,
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ),
        metadata={"option_type": "CE"},
    )

    with pytest.raises(OrderValidationError, match="MARKET orders"):
        validate_order_price(order, {"ltp": 100.0, "bid": 99.9, "ask": 100.1})


def test_option_selling_entry_rejects_wide_spread() -> None:
    order = OrderRequest(
        symbol="NIFTY26000PE",
        side=OrderSide.SELL,
        quantity=75,
        order_type=OrderType.LIMIT,
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ),
        limit_price=99.5,
        metadata={"option_type": "PE"},
    )

    with pytest.raises(OrderValidationError, match="Bid-ask spread"):
        validate_order_price(order, {"ltp": 100.0, "bid": 98.0, "ask": 102.0})


def test_stop_loss_timeout_converts_to_emergency_market_when_allowed() -> None:
    validator = OrderValidator()
    order = build_short_option_stop_loss_order(
        symbol="NIFTY26000CE",
        quantity=75,
        entry_price=100.0,
        stop_loss_percent=25.0,
        buffer=1.0,
    )
    submitted_at = datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ)

    emergency = validator.emergency_market_exit(
        order,
        submitted_at=submitted_at,
        now=submitted_at + timedelta(seconds=31),
    )

    assert emergency is not None
    assert emergency.order_type == OrderType.MARKET
    assert emergency.metadata["emergency_exit"] is True
    assert emergency.metadata["short_option_exit"] is True


def test_market_order_allowed_only_when_marked_emergency_exit() -> None:
    order = OrderRequest(
        symbol="NIFTY26000CE",
        side=OrderSide.BUY,
        quantity=75,
        order_type=OrderType.MARKET,
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ),
        metadata={"emergency_exit": True, "short_option_exit": True},
    )

    assert validate_order_price(order, {"ltp": 130.0, "bid": 129.9, "ask": 130.1}).approved is True
