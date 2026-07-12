"""Focused tests for shared PAPER runtime models."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from models import (
    MarketSnapshot,
    Mode,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperFill,
    PaperOrder,
    PaperPortfolioState,
    PaperPosition,
    ReconciliationResult,
    StrategyLeg,
    StrategyType,
    SystemHealthState,
    TradingCycleResult,
)
from utils.time_utils import KOLKATA_TZ_NAME


def _runtime_models() -> tuple[PaperFill, PaperOrder, PaperPosition, PaperPortfolioState, SystemHealthState, ReconciliationResult]:
    now = datetime(2026, 7, 12, 4, 30, tzinfo=timezone.utc)
    fill = PaperFill("fill-1", "order-1", now, "NIFTYCE", "101", OrderSide.BUY, 75, 75, 100.0, 12.0, 1.0)
    order = PaperOrder(
        "order-1", now, now, "NIFTYCE", "101", "strategy-1", "leg-1",
        OrderSide.BUY, OrderType.LIMIT, 75, 75, OrderStatus.FILLED,
        requested_price=101.0, fill_price=100.0, filled_quantity=75, fills=(fill,),
    )
    leg = StrategyLeg("NIFTYCE", OrderSide.BUY, 75, OrderType.LIMIT, limit_price=101.0)
    position = PaperPosition(
        "position-1", now, now, "NIFTY", "100", "strategy-1", StrategyType.LONG_GAMMA,
        (leg,), OrderSide.BUY, 75, 75, OrderStatus.OPEN, 100.0, 102.0, 5000.0,
        "approved paper entry", unrealized_pnl=150.0, transaction_costs=12.0, slippage=1.0,
    )
    portfolio = PaperPortfolioState(
        "paper-main", now, 1_000_000.0, 1_000_138.0, 900_000.0, 100_000.0,
        5000.0, unrealized_pnl=150.0, transaction_costs=12.0, positions=(position,),
    )
    health = SystemHealthState("health-1", now, True, True, True, False)
    reconciliation = ReconciliationResult("recon-1", now, True, 1, 1, 1)
    return fill, order, position, portfolio, health, reconciliation


def test_all_paper_runtime_models_can_be_created_in_ist() -> None:
    fill, order, position, portfolio, health, reconciliation = _runtime_models()
    snapshot = MarketSnapshot("snapshot-1", fill.timestamp, "NIFTY", "100", 25000.0, 24999.0, 25001.0, 1000)
    result = TradingCycleResult(
        "cycle-1", fill.timestamp, fill.timestamp, True, "NO_TRADE", portfolio, health,
        reconciliation, (order,), (fill,),
    )
    assert all(
        value.tzinfo is not None and getattr(value.tzinfo, "key", None) == KOLKATA_TZ_NAME
        for value in (fill.timestamp, order.created_at, position.opened_at, portfolio.timestamp, snapshot.timestamp, result.started_at)
    )
    assert portfolio.mode is Mode.PAPER


def test_model_validation_rejects_unsafe_or_invalid_values() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="quantity must be positive"):
        PaperFill("f", "o", now, "NIFTY", "1", OrderSide.BUY, 0, 75, 10.0)
    with pytest.raises(ValueError, match="bid cannot exceed ask"):
        MarketSnapshot("s", now, "NIFTY", "1", 100.0, 102.0, 101.0)
    with pytest.raises(ValueError, match="mode must remain PAPER"):
        PaperPortfolioState("p", now, 100.0, 100.0, 100.0, 0.0, 0.0, mode=Mode.LIVE)


def test_enum_serialization_is_json_safe() -> None:
    fill, order, *_ = _runtime_models()
    payload = order.to_dict()
    assert payload["side"] == "BUY"
    assert payload["order_type"] == "LIMIT"
    assert payload["status"] == "FILLED"
    assert payload["fills"][0]["fill_id"] == fill.fill_id
    json.dumps(payload)


def test_trading_cycle_round_trip_persistence() -> None:
    fill, order, position, portfolio, health, reconciliation = _runtime_models()
    original = TradingCycleResult(
        "cycle-1", fill.timestamp, fill.timestamp, True, "POSITION_OPEN", portfolio, health,
        reconciliation, (order,), (fill,), metadata={"source": "deterministic_test"},
    )
    restored = TradingCycleResult.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.portfolio.positions[0] == position
    assert restored.orders[0].side is OrderSide.BUY
    assert restored.portfolio.mode is Mode.PAPER
