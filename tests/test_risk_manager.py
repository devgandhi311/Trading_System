"""Tests for risk guards and RiskManager orchestration."""

from __future__ import annotations

from datetime import datetime

from models import OrderSide, OrderType, PredictionResult, RegimeResult, RegimeType, StrategyLeg, StrategyPlan, StrategyType
from risk import KillSwitch, RiskManager, SlippageModel
from utils.time_utils import KOLKATA_TZ


MARKET_TIME = datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ)
AFTER_CUTOFF = datetime(2026, 1, 1, 15, 10, tzinfo=KOLKATA_TZ)


def _plan(max_loss: float = 2_000.0) -> StrategyPlan:
    return StrategyPlan(
        strategy_id="risk-test",
        strategy_type=StrategyType.CREDIT_SPREAD,
        symbol="NIFTY",
        created_at=MARKET_TIME,
        legs=(
            StrategyLeg(
                symbol="NIFTY26000PE",
                side=OrderSide.SELL,
                quantity=75,
                order_type=OrderType.LIMIT,
                limit_price=100.0,
            ),
            StrategyLeg(
                symbol="NIFTY25900PE",
                side=OrderSide.BUY,
                quantity=75,
                order_type=OrderType.LIMIT,
                limit_price=80.0,
            ),
        ),
        confidence=0.8,
        rationale="test plan",
        max_loss=max_loss,
        expected_profit=1_500.0,
        metadata={"margin_estimate_placeholder": max_loss},
    )


def _account(**overrides: float | int) -> dict[str, float | int]:
    account: dict[str, float | int] = {
        "daily_pnl": 0.0,
        "peak_daily_pnl": 0.0,
        "trades_today": 0,
        "open_positions": 0,
        "consecutive_losses": 0,
        "margin_used": 0.0,
    }
    account.update(overrides)
    return account


def test_risk_manager_approves_valid_plan() -> None:
    decision = RiskManager().validate(
        _plan(),
        account_context=_account(),
        market_context={"bid_ask_spread_percent": 0.2, "slippage_percent": 0.1},
        timestamp=MARKET_TIME,
    )

    assert decision.approved is True
    assert decision.rejection_reason is None
    assert decision.allowed_quantity == 75
    assert decision.max_risk_amount == 5_000.0
    assert decision.risk_used == 2_000.0
    assert all(decision.checks.values())


def test_risk_manager_rejects_daily_loss_limit() -> None:
    decision = RiskManager().validate(
        _plan(),
        account_context=_account(daily_pnl=-20_000.0),
        timestamp=MARKET_TIME,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Daily max loss limit hit."
    assert decision.checks["daily_max_loss"] is False


def test_risk_manager_rejects_per_trade_loss_limit() -> None:
    decision = RiskManager().validate(
        _plan(max_loss=6_000.0),
        account_context=_account(),
        timestamp=MARKET_TIME,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Per-trade max loss exceeded."
    assert decision.checks["per_trade_max_loss"] is False


def test_risk_manager_rejects_trade_count_limit() -> None:
    decision = RiskManager().validate(
        _plan(),
        account_context=_account(trades_today=10),
        timestamp=MARKET_TIME,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Max trades per day reached."
    assert decision.checks["max_trades_per_day"] is False


def test_risk_manager_rejects_spread_and_slippage_limits() -> None:
    spread_decision = RiskManager().validate(
        _plan(),
        account_context=_account(),
        market_context={"bid_ask_spread_percent": 1.5, "slippage_percent": 0.1},
        timestamp=MARKET_TIME,
    )
    slippage_decision = RiskManager().validate(
        _plan(),
        account_context=_account(),
        market_context={"bid_ask_spread_percent": 0.2, "slippage_percent": 0.3},
        timestamp=MARKET_TIME,
    )

    assert spread_decision.rejection_reason == "Bid-ask spread limit exceeded."
    assert spread_decision.checks["bid_ask_spread_limit"] is False
    assert slippage_decision.rejection_reason == "Slippage limit exceeded."
    assert slippage_decision.checks["slippage_limit"] is False


def test_risk_manager_rejects_event_risk_and_time_cutoff() -> None:
    event_regime = RegimeResult(
        symbol="NIFTY",
        timestamp=MARKET_TIME,
        regime=RegimeType.UNKNOWN,
        confidence=0.8,
        lookback_bars=60,
        regime_name="Event-risk no-trade",
        event_risk_score=0.9,
        allowed_to_trade=False,
    )
    event_decision = RiskManager().validate(_plan(), account_context=_account(), regime=event_regime, timestamp=MARKET_TIME)
    cutoff_decision = RiskManager().validate(_plan(), account_context=_account(), timestamp=AFTER_CUTOFF)

    assert event_decision.rejection_reason == "Event-risk regime blocks new trades."
    assert event_decision.checks["event_risk_regime"] is False
    assert cutoff_decision.rejection_reason == "No new trade allowed after configured cutoff."
    assert cutoff_decision.checks["no_new_trade_after"] is False


def test_risk_manager_rejects_active_kill_switch() -> None:
    kill_switch = KillSwitch()
    kill_switch.trigger("manual stop")

    decision = RiskManager(kill_switch=kill_switch).validate(
        _plan(),
        account_context=_account(),
        timestamp=MARKET_TIME,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Kill switch is active."
    assert decision.checks["kill_switch_inactive"] is False


def test_slippage_model_can_estimate_from_leg_metadata() -> None:
    plan = StrategyPlan(
        strategy_id="spread-estimate",
        strategy_type=StrategyType.CUSTOM,
        symbol="NIFTY",
        created_at=MARKET_TIME,
        legs=(
            StrategyLeg(
                symbol="NIFTY26000CE",
                side=OrderSide.BUY,
                quantity=75,
                order_type=OrderType.LIMIT,
                limit_price=101.0,
                metadata={"bid": 100.0, "ask": 101.0},
            ),
        ),
        confidence=0.8,
        rationale="test",
        max_loss=1_000.0,
    )

    result = SlippageModel(max_bid_ask_spread_percent=1.0, max_slippage_percent=0.6).check(plan)

    assert result.approved is True
    assert 0.99 < result.max_spread_percent < 1.0
