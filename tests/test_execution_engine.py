"""Tests for execution engine orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta

from execution import ExecutionEngine, ExecutionTradeState, OrderRouter
from models import OrderResponse, OrderSide, OrderStatus, OrderType, StrategyLeg, StrategyPlan, StrategyType
from models.order_models import OrderRequest
from utils.time_utils import KOLKATA_TZ


MARKET_TIME = datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ)


class FakeOrderManager:
    """OrderManager test double."""

    def __init__(self, statuses: list[OrderStatus] | None = None) -> None:
        self.statuses = statuses or []
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []
        self.counter = 0

    def place_order(self, order_request: OrderRequest) -> OrderResponse:
        self.placed.append(order_request)
        self.counter += 1
        status = self.statuses.pop(0) if self.statuses else OrderStatus.OPEN
        filled_quantity = order_request.quantity if status == OrderStatus.FILLED else 0
        average_price = order_request.limit_price if status == OrderStatus.FILLED else None
        return OrderResponse(
            order_id=f"fake-{self.counter}",
            status=status,
            timestamp=MARKET_TIME,
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            filled_quantity=filled_quantity,
            average_price=average_price,
        )

    def cancel_order(self, order_id: str) -> OrderResponse:
        self.cancelled.append(order_id)
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            timestamp=MARKET_TIME,
            symbol="NIFTY26000CE",
            side=OrderSide.SELL,
            quantity=75,
        )


def _quote(symbol: str) -> dict[str, float]:
    return {"ltp": 100.0, "bid": 99.95, "ask": 100.05}


def _plan(max_loss: float = 2_000.0) -> StrategyPlan:
    return StrategyPlan(
        strategy_id="exec-test",
        strategy_type=StrategyType.CREDIT_SPREAD,
        symbol="NIFTY",
        created_at=MARKET_TIME,
        legs=(
            StrategyLeg(
                symbol="NIFTY26000CE",
                side=OrderSide.SELL,
                quantity=75,
                order_type=OrderType.LIMIT,
                limit_price=99.95,
                metadata={"option_type": "CE", "bid": 99.95, "ask": 100.05},
            ),
        ),
        confidence=0.8,
        rationale="test",
        max_loss=max_loss,
        expected_profit=1_000.0,
        metadata={
            "margin_estimate_placeholder": max_loss,
            "stop_loss_percent": 0.0,
            "sl_limit_buffer": 0.1,
            "target_price_buffer": 0.0,
            "target_profit": 500.0,
            "stop_loss": 300.0,
        },
    )


def _account(**overrides: float | int) -> dict[str, float | int]:
    account: dict[str, float | int] = {
        "daily_pnl": 0.0,
        "peak_daily_pnl": 0.0,
        "trades_today": 0,
        "open_positions": 0,
        "consecutive_losses": 0,
        "margin_used": 0.0,
        "available_margin": 100_000.0,
        "initial_capital": 1_000_000,
    }
    account.update(overrides)
    return account


def _engine(fake_order_manager: FakeOrderManager) -> ExecutionEngine:
    router = OrderRouter(order_manager=fake_order_manager)
    return ExecutionEngine(order_router=router)


def test_execution_engine_places_entries_and_protective_orders() -> None:
    fake = FakeOrderManager(statuses=[OrderStatus.FILLED, OrderStatus.OPEN, OrderStatus.OPEN])
    result = _engine(fake).execute_strategy(
        _plan(),
        quote_provider=_quote,
        account_context=_account(),
        market_context={"bid_ask_spread_percent": 0.1, "slippage_percent": 0.1},
        timestamp=MARKET_TIME,
    )

    assert result.approved is True
    assert result.risk_decision is not None
    assert result.risk_decision.approved is True
    assert len(result.entry_orders) == 1
    assert len(result.stop_loss_orders) == 1
    assert len(result.profit_booking_orders) == 1
    assert fake.placed[0].order_type == OrderType.LIMIT
    assert fake.placed[1].order_type == OrderType.SL_LIMIT
    assert fake.placed[2].order_type == OrderType.LIMIT


def test_execution_engine_rejects_insufficient_margin() -> None:
    fake = FakeOrderManager()
    result = _engine(fake).execute_strategy(
        _plan(),
        quote_provider=_quote,
        account_context=_account(available_margin=100.0),
        market_context={"bid_ask_spread_percent": 0.1, "slippage_percent": 0.1},
        timestamp=MARKET_TIME,
    )

    assert result.approved is False
    assert result.status == "margin_rejected"
    assert fake.placed == []


def test_execution_engine_cancels_unfilled_after_timeout() -> None:
    fake = FakeOrderManager(statuses=[OrderStatus.OPEN])
    engine = _engine(fake)
    basket = engine.order_router.prepare_entry_basket(_plan())
    responses = engine.order_router.place_basket(basket, _quote)
    cancelled = engine.cancel_unfilled_after_timeout(
        basket,
        responses,
        submitted_at=MARKET_TIME,
        now=MARKET_TIME + timedelta(seconds=31),
    )

    assert len(cancelled) == 1
    assert cancelled[0].status == OrderStatus.CANCELLED
    assert fake.cancelled == ["fake-1"]


def test_execution_engine_squareoffs_partial_fill() -> None:
    fake = FakeOrderManager(statuses=[OrderStatus.FILLED, OrderStatus.OPEN, OrderStatus.FILLED])
    plan = StrategyPlan(
        **{
            **_plan().__dict__,
            "legs": (
                _plan().legs[0],
                StrategyLeg(
                    symbol="NIFTY26000PE",
                    side=OrderSide.BUY,
                    quantity=75,
                    order_type=OrderType.LIMIT,
                    limit_price=100.05,
                    metadata={"option_type": "PE", "bid": 99.95, "ask": 100.05},
                ),
            ),
        }
    )

    result = _engine(fake).execute_strategy(
        plan,
        quote_provider=_quote,
        account_context=_account(),
        market_context={"bid_ask_spread_percent": 0.1, "slippage_percent": 0.1},
        timestamp=MARKET_TIME,
    )

    assert result.squareoff_result is not None
    assert result.squareoff_result.reason == "partial_fill_squareoff"
    assert result.squareoff_result.orders[0].side == OrderSide.BUY


def test_execution_engine_exit_conditions_cover_target_daily_loss_and_manual() -> None:
    engine = _engine(FakeOrderManager())
    state = ExecutionTradeState.from_plan(_plan())
    state.unrealized_pnl = 600.0

    assert engine.evaluate_exit_conditions(_plan(), state, timestamp=MARKET_TIME) == "target"
    assert engine.evaluate_exit_conditions(_plan(), state, account_context={"daily_pnl": -20_000}, timestamp=MARKET_TIME) == "daily_loss"
    assert engine.evaluate_exit_conditions(_plan(), state, manual_squareoff=True, timestamp=MARKET_TIME) == "manual_squareoff"
