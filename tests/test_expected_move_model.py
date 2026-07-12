"""Tests for expected move model."""

from __future__ import annotations

import numpy as np

from quant.distribution import ExpectedMoveModel


def _returns() -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.normal(loc=0.0003, scale=0.01, size=260)


def test_expected_move_from_realized_vol_and_iv() -> None:
    model = ExpectedMoveModel()

    realized = model.calculate_from_realized_vol(
        spot_price=26000.0,
        realized_volatility=0.18,
        days_to_expiry=7,
    )
    iv = model.calculate_from_iv(
        spot_price=26000.0,
        implied_volatility=20.0,
        days_to_expiry=7,
    )

    assert realized.expected_move_points > 0
    assert iv.expected_move_points > realized.expected_move_points
    assert iv.lower_price < 26000.0 < iv.upper_price
    assert iv.method_used == "iv"


def test_expected_move_empirical_and_monte_carlo() -> None:
    model = ExpectedMoveModel()

    empirical = model.calculate_from_empirical_returns(
        spot_price=26000.0,
        historical_returns=_returns(),
        days_to_expiry=5,
    )
    monte_carlo = model.calculate_from_monte_carlo(
        spot_price=26000.0,
        historical_returns=_returns(),
        days_to_expiry=5,
        realized_volatility=0.18,
        simulations=1000,
        seed=4,
    )

    assert empirical.expected_move_points > 0
    assert monte_carlo.expected_move_points > 0
    assert empirical.confidence > 0
    assert monte_carlo.method_used == "monte_carlo"


def test_blended_expected_move_applies_news_and_regime_multiplier() -> None:
    model = ExpectedMoveModel()
    returns = _returns()

    low_risk = model.blend_expected_move(
        spot_price=26000.0,
        days_to_expiry=7,
        historical_returns=returns,
        realized_volatility=0.18,
        implied_volatility=0.20,
        regime_result={"volatility_score": 0.2},
        news_event_risk={"event_risk_level": "LOW"},
    )
    high_risk = model.blend_expected_move(
        spot_price=26000.0,
        days_to_expiry=7,
        historical_returns=returns,
        realized_volatility=0.18,
        implied_volatility=0.20,
        regime_result={"volatility_score": 0.9},
        news_event_risk={"event_risk_level": "HIGH"},
    )

    assert high_risk.expected_move_points > low_risk.expected_move_points
    assert high_risk.method_used.startswith("blend:")
    assert 0.0 <= high_risk.confidence <= 1.0
