"""Tests for dynamic hedge recommendation engine."""

from __future__ import annotations

from risk import DynamicHedgeEngine


def test_dynamic_hedge_recommends_delta_hedge() -> None:
    rec = DynamicHedgeEngine(delta_threshold=0.25).recommend(
        current_positions=({"symbol": "NIFTY", "underlying": "NIFTY"},),
        greeks={"delta": 0.45, "gamma": 0.02},
        spot_move=0.01,
        volatility_move=0.02,
        news_risk={"event_risk_level": "LOW"},
        time_to_expiry=7,
    )

    assert rec.hedge_required is True
    assert "delta exceeds threshold" in rec.hedge_reason
    assert rec.hedge_legs[0]["action"] == "SELL_UNDERLYING"


def test_dynamic_hedge_recommends_critical_news_reduction() -> None:
    rec = DynamicHedgeEngine().recommend(
        current_positions=({"symbol": "NIFTY"},),
        greeks={"delta": 0.05, "gamma": 0.01},
        news_risk={"event_risk_level": "EXTREME"},
        time_to_expiry=5,
    )

    assert rec.hedge_required is True
    assert rec.urgency == "CRITICAL"
    assert any(leg["action"] == "REDUCE_POSITION" for leg in rec.hedge_legs)


def test_dynamic_hedge_no_action_when_within_thresholds() -> None:
    rec = DynamicHedgeEngine().recommend(
        current_positions=(),
        greeks={"delta": 0.01, "gamma": 0.01},
        news_risk={"event_risk_level": "LOW"},
        time_to_expiry=10,
    )

    assert rec.hedge_required is False
    assert rec.urgency == "NONE"
    assert rec.hedge_legs == ()
