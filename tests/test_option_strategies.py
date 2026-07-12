"""Tests for option strategy plan builders."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from models import OptionChainSnapshot, OptionContract, OrderSide, PredictionResult, RegimeType, StrategyType
from strategies import (
    CreditSpreadStrategy,
    DebitSpreadStrategy,
    ExpectedMoveCreditSpreadStrategy,
    IronCondorStrategy,
    LongGammaStrategy,
    ShortStrangleStrategy,
)


def _chain() -> OptionChainSnapshot:
    expiry = date.today() + timedelta(days=10)
    contracts = []
    for strike, call_price, put_price in (
        (25700.0, 340.0, 5.0),
        (25800.0, 260.0, 40.0),
        (25900.0, 180.0, 70.0),
        (26000.0, 110.0, 110.0),
        (26100.0, 70.0, 180.0),
        (26200.0, 40.0, 260.0),
    ):
        contracts.extend(
            (
                OptionContract(
                    symbol=f"NIFTY{int(strike)}CE",
                    underlying="NIFTY",
                    expiry=expiry,
                    strike=strike,
                    option_type="CE",
                    lot_size=75,
                    bid=call_price - 1.0,
                    ask=call_price + 1.0,
                    last_price=call_price,
                ),
                OptionContract(
                    symbol=f"NIFTY{int(strike)}PE",
                    underlying="NIFTY",
                    expiry=expiry,
                    strike=strike,
                    option_type="PE",
                    lot_size=75,
                    bid=put_price - 1.0,
                    ask=put_price + 1.0,
                    last_price=put_price,
                ),
            )
        )
    return OptionChainSnapshot(
        underlying="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 30),
        spot_price=26000.0,
        expiry=expiry,
        contracts=tuple(contracts),
    )


def _prediction(regime: RegimeType = RegimeType.TRENDING) -> PredictionResult:
    return PredictionResult(
        symbol="NIFTY",
        underlying="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 30),
        predicted_direction="bullish",
        confidence=0.82,
        confidence_score=0.82,
        horizon_minutes=375,
        regime=regime,
        bullish_probability=0.62,
        bearish_probability=0.18,
        sideways_probability=0.2,
        expected_move_percent=1.2,
        expected_return_percent=0.528,
        trade_allowed=True,
        reason="test prediction",
    )


def test_iron_condor_builds_four_defined_risk_legs() -> None:
    plan = IronCondorStrategy().build_plan(
        _prediction(RegimeType.RANGE_BOUND),
        _chain(),
        lot_size=75,
        max_lots=1,
        liquidity_score=0.8,
    )

    assert plan.strategy_type == StrategyType.IRON_CONDOR
    assert len(plan.legs) == 4
    assert [leg.side for leg in plan.legs] == [OrderSide.SELL, OrderSide.BUY, OrderSide.SELL, OrderSide.BUY]
    assert plan.metadata["max_profit"] > 0
    assert plan.metadata["max_loss"] >= 0
    assert len(plan.metadata["breakeven"]) == 2


def test_credit_spread_builds_bull_put_spread() -> None:
    plan = CreditSpreadStrategy(variant="bull_put").build_plan(
        _prediction(RegimeType.TRENDING),
        _chain(),
        lot_size=75,
        max_lots=1,
        liquidity_score=0.8,
    )

    assert plan.strategy_type == StrategyType.CREDIT_SPREAD
    assert len(plan.legs) == 2
    assert plan.legs[0].side == OrderSide.SELL
    assert plan.legs[1].side == OrderSide.BUY
    assert plan.metadata["variant"] == "bull_put"
    assert plan.metadata["target_profit"] == plan.metadata["max_profit"] * 0.5


def test_expected_move_credit_spread_uses_probability_and_defined_risk() -> None:
    plan = ExpectedMoveCreditSpreadStrategy().build_plan(
        _prediction(RegimeType.TRENDING),
        _chain(),
        lot_size=75,
        max_lots=1,
        liquidity_score=0.8,
    )

    assert plan.strategy_type == StrategyType.EXPECTED_MOVE_CREDIT_SPREAD
    assert len(plan.legs) == 2
    assert plan.legs[0].side == OrderSide.SELL
    assert plan.legs[1].side == OrderSide.BUY
    assert plan.metadata["valid"] is True
    assert plan.metadata["expected_value"] >= 0
    assert plan.metadata["probability_of_profit"] >= 0.68
    assert plan.metadata["max_loss"] > 0
    assert plan.metadata["spread_width"] <= 500


def test_debit_spread_builds_bull_call_spread() -> None:
    plan = DebitSpreadStrategy(variant="bull_call").build_plan(
        _prediction(RegimeType.TRENDING),
        _chain(),
        lot_size=75,
        max_lots=1,
        liquidity_score=0.8,
    )

    assert plan.strategy_type == StrategyType.DEBIT_SPREAD
    assert len(plan.legs) == 2
    assert plan.legs[0].side == OrderSide.BUY
    assert plan.legs[1].side == OrderSide.SELL
    assert plan.metadata["variant"] == "bull_call"
    assert plan.max_loss == plan.metadata["max_loss"]


def test_long_gamma_builds_straddle() -> None:
    plan = LongGammaStrategy(variant="long_straddle").build_plan(
        _prediction(RegimeType.HIGH_VOLATILITY),
        _chain(),
        lot_size=75,
        max_lots=1,
        liquidity_score=0.8,
    )

    assert plan.strategy_type == StrategyType.LONG_GAMMA
    assert len(plan.legs) == 2
    assert all(leg.side == OrderSide.BUY for leg in plan.legs)
    assert plan.metadata["variant"] == "long_straddle"
    assert plan.metadata["max_loss"] > 0


def test_short_strangle_is_disabled_by_default() -> None:
    plan = ShortStrangleStrategy().build_plan(
        _prediction(RegimeType.RANGE_BOUND),
        _chain(),
        lot_size=75,
        max_lots=1,
        liquidity_score=0.8,
    )

    assert plan.strategy_type == StrategyType.SHORT_STRANGLE
    assert plan.legs == ()
    assert plan.metadata["valid"] is False
    assert plan.metadata["enabled"] is False
    assert "disabled" in plan.rationale.lower()


def test_enabled_short_strangle_includes_emergency_hedge_logic() -> None:
    plan = ShortStrangleStrategy(enabled=True).build_plan(
        _prediction(RegimeType.RANGE_BOUND),
        _chain(),
        lot_size=75,
        max_lots=1,
        liquidity_score=0.9,
    )

    assert len(plan.legs) == 2
    assert all(leg.side == OrderSide.SELL for leg in plan.legs)
    assert plan.metadata["emergency_hedge_logic"]["enabled"] is True
    assert plan.metadata["emergency_hedge_logic"]["hedge_legs"]
    assert plan.metadata["stop_loss"] == plan.metadata["max_loss"]
