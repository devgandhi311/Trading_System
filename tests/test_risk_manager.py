"""Tests for risk guards and RiskManager orchestration."""

from __future__ import annotations

from datetime import datetime

from models import OrderSide, OrderType, PredictionResult, RegimeResult, RegimeType, StrategyLeg, StrategyPlan, StrategyType
from risk import KillSwitch, PositionSizer, RiskManager, SlippageModel
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
        account_context=_account(initial_capital=1_000_000),
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
        account_context=_account(initial_capital=1_000_000, daily_pnl=-20_000.0),
        timestamp=MARKET_TIME,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Daily max loss limit hit."
    assert decision.checks["daily_max_loss"] is False


def test_risk_manager_rejects_per_trade_loss_limit() -> None:
    decision = RiskManager().validate(
        _plan(max_loss=6_000.0),
        account_context=_account(initial_capital=1_000_000),
        timestamp=MARKET_TIME,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Per-trade max loss exceeded."
    assert decision.checks["per_trade_max_loss"] is False


def test_risk_manager_rejects_trade_count_limit() -> None:
    decision = RiskManager().validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000, trades_today=10),
        timestamp=MARKET_TIME,
    )

    assert decision.approved is False
    assert decision.rejection_reason == "Max trades per day reached."
    assert decision.checks["max_trades_per_day"] is False


def test_risk_manager_rejects_spread_and_slippage_limits() -> None:
    spread_decision = RiskManager().validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000),
        market_context={"bid_ask_spread_percent": 1.5, "slippage_percent": 0.1},
        timestamp=MARKET_TIME,
    )
    slippage_decision = RiskManager().validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000),
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
    event_decision = RiskManager().validate(_plan(), account_context=_account(initial_capital=1_000_000), regime=event_regime, timestamp=MARKET_TIME)
    cutoff_decision = RiskManager().validate(_plan(), account_context=_account(initial_capital=1_000_000), timestamp=AFTER_CUTOFF)

    assert event_decision.rejection_reason == "Event-risk regime blocks new trades."
    assert event_decision.checks["event_risk_regime"] is False
    assert cutoff_decision.rejection_reason == "No new trade allowed after configured cutoff."
    assert cutoff_decision.checks["no_new_trade_after"] is False


def test_risk_manager_rejects_active_kill_switch() -> None:
    kill_switch = KillSwitch()
    kill_switch.trigger("manual stop")

    decision = RiskManager(kill_switch=kill_switch).validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000),
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


def test_sizing_uses_current_equity_after_profit_and_loss() -> None:
    manager = RiskManager()
    profit = manager.validate(_plan(max_loss=5_500), account_context=_account(initial_capital=1_000_000, realized_pnl=200_000), timestamp=MARKET_TIME)
    loss = manager.validate(_plan(max_loss=4_000), account_context=_account(initial_capital=1_000_000, realized_pnl=-400_000), timestamp=MARKET_TIME)
    assert profit.approved is True
    assert profit.max_risk_amount == 6_000
    assert loss.approved is False
    assert loss.max_risk_amount == 3_000


def test_zero_and_negative_equity_fail_closed() -> None:
    manager = RiskManager()
    zero = manager.validate(_plan(), account_context=_account(initial_capital=1_000, realized_pnl=-1_000), timestamp=MARKET_TIME)
    negative = manager.validate(_plan(), account_context=_account(current_equity=-1), timestamp=MARKET_TIME)
    assert zero.rejection_reason == "Current equity is zero or negative."
    assert negative.rejection_reason == "Current equity is zero or negative."


def test_pending_and_partial_fills_reserve_risk() -> None:
    manager = RiskManager()
    pending = manager.validate(_plan(), account_context=_account(initial_capital=1_000_000, pending_order_risk=4_000), timestamp=MARKET_TIME)
    partial = manager.validate(_plan(), account_context=_account(initial_capital=1_000_000, partial_fill_risk=4_000), timestamp=MARKET_TIME)
    assert pending.approved is False and pending.max_risk_amount == 1_000
    assert partial.approved is False and partial.max_risk_amount == 1_000


def test_pending_and_partial_margin_are_reserved() -> None:
    decision = RiskManager().validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000, margin_used=480_000, pending_order_margin=15_000, partial_fill_margin=10_000),
        timestamp=MARKET_TIME,
    )
    assert decision.rejection_reason == "Max margin usage exceeded."


def test_explicit_current_equity_wins() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000, realized_pnl=0, current_equity=2_000_000),
        timestamp=MARKET_TIME,
    )
    assert decision.approved is True
    assert decision.metadata["current_equity"] == 2_000_000
    assert decision.metadata["current_equity_source"] == "explicit"
    assert decision.metadata["final_max_risk_amount"] == 10_000.0


def test_derived_equity_after_profit() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(max_loss=5_500),
        account_context=_account(initial_capital=1_000_000, realized_pnl=200_000),
        timestamp=MARKET_TIME,
    )
    assert decision.approved is True
    assert decision.metadata["derived_current_equity"] == 1_200_000.0
    assert decision.metadata["current_equity_source"] == "derived"
    assert decision.max_risk_amount == 6_000.0


def test_derived_equity_after_loss() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(max_loss=4_000),
        account_context=_account(initial_capital=1_000_000, realized_pnl=-400_000),
        timestamp=MARKET_TIME,
    )
    assert decision.approved is False
    assert decision.metadata["derived_current_equity"] == 600_000.0
    assert decision.metadata["current_equity_source"] == "derived"
    assert decision.rejection_reason == "Per-trade max loss exceeded."
    assert decision.max_risk_amount == 3_000.0


def test_missing_account_context_fails_closed() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(),
        account_context=None,
        timestamp=MARKET_TIME,
    )
    assert decision.approved is False
    assert decision.rejection_reason == "Missing authoritative account state."
    assert decision.metadata["current_equity_source"] == "missing"


def test_costs_reduce_equity() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000, realized_pnl=0, accrued_costs=50_000),
        timestamp=MARKET_TIME,
    )
    assert decision.approved is True
    assert decision.metadata["derived_current_equity"] == 950_000.0
    assert decision.max_risk_amount == 4_750.0


def test_slippage_reduces_equity() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000, realized_pnl=0, accrued_costs=20_000),
        timestamp=MARKET_TIME,
    )
    assert decision.approved is True
    assert decision.metadata["derived_current_equity"] == 980_000.0
    assert decision.max_risk_amount == 4_900.0


def test_percentage_cap() -> None:
    sizer = PositionSizer(capital=1_000_000, max_loss_per_trade_percent=0.5)
    result = sizer.size(_plan())
    assert result.max_risk_amount == 5_000.0


def test_absolute_cap() -> None:
    sizer = PositionSizer(capital=1_000_000, max_loss_per_trade_percent=0.5, max_loss_per_trade_absolute=3_000.0)
    result = sizer.size(_plan())
    assert result.max_risk_amount == 3_000.0


def test_combined_cap_behavior() -> None:
    sizer = PositionSizer(capital=1_000_000, max_loss_per_trade_percent=0.5, max_loss_per_trade_absolute=3_000.0)
    result = sizer.size(_plan(max_loss=2_000.0))
    assert result.max_risk_amount == 3_000.0
    assert result.approved is True
    assert result.allowed_quantity == 75


def test_risk_metadata_is_complete() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000),
        timestamp=MARKET_TIME,
    )
    assert decision.approved is True
    meta = decision.metadata
    assert "current_equity_source" in meta
    assert "derived_current_equity" in meta
    assert "percentage_risk_limit" in meta
    assert "absolute_cap" in meta
    assert "final_max_risk_amount" in meta
    assert "remaining_risk" in meta
    assert "reserved_risk" in meta


def test_current_equity_source_is_recorded() -> None:
    manager = RiskManager()
    explicit = manager.validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000, current_equity=1_500_000),
        timestamp=MARKET_TIME,
    )
    assert explicit.metadata["current_equity_source"] == "explicit"

    derived = manager.validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000, realized_pnl=200_000),
        timestamp=MARKET_TIME,
    )
    assert derived.metadata["current_equity_source"] == "derived"


def test_existing_valid_plan_remains_approved() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(max_loss=2_000),
        account_context=_account(initial_capital=1_000_000),
        timestamp=MARKET_TIME,
    )
    assert decision.approved is True
    assert decision.allowed_quantity == 75


def test_no_fallback_after_explicit_zero_equity() -> None:
    manager = RiskManager()
    decision = manager.validate(
        _plan(),
        account_context=_account(initial_capital=1_000_000, current_equity=0),
        timestamp=MARKET_TIME,
    )
    assert decision.approved is False
    assert decision.rejection_reason == "Current equity is zero or negative."
