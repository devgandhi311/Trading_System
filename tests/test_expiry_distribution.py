"""Tests for quant expiry distribution estimation."""

from __future__ import annotations

import numpy as np

from quant.distribution import ExpiryDistributionEngine


def _returns() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(loc=0.0004, scale=0.01, size=300)


def test_expiry_distribution_outputs_bounds_and_probabilities() -> None:
    engine = ExpiryDistributionEngine(monte_carlo_simulations=1000, random_seed=11)

    result = engine.estimate(
        spot_price=26000.0,
        historical_returns=_returns(),
        days_to_expiry=7,
        realized_volatility=0.18,
        predicted_volatility=0.20,
        regime_result={"event_risk_score": 0.0},
        news_event_risk={"event_risk_level": "LOW"},
    )

    assert result.expected_expiry_price > 0
    assert result.expected_move_points > 0
    assert result.expected_move_percent > 0
    assert result.lower_bound_68 < result.upper_bound_68
    assert result.lower_bound_95 < result.lower_bound_68
    assert result.upper_bound_95 > result.upper_bound_68
    assert 0.0 <= result.probability_below_strike(26000.0) <= 1.0
    assert 0.0 <= result.probability_above_strike(26000.0) <= 1.0
    assert np.isclose(
        result.probability_below_strike(26000.0) + result.probability_above_strike(26000.0),
        1.0,
    )
    assert result.probability_between(result.lower_bound_68, result.upper_bound_68) > 0.5
    assert result.tail_risk_score >= 0.0


def test_parametric_distribution_fallback_without_samples() -> None:
    engine = ExpiryDistributionEngine(use_empirical=False, use_monte_carlo=False, use_parametric=True)

    result = engine.estimate(
        spot_price=100.0,
        historical_returns=[0.001, -0.002, 0.003, 0.0],
        days_to_expiry=5,
        realized_volatility=20.0,
    )

    assert result.model_used == "parametric"
    assert result.samples == ()
    assert result.parametric_distribution is not None
    assert result.lower_bound_90 < result.expected_expiry_price < result.upper_bound_90
    assert 0.0 < result.probability_below_strike(100.0) < 1.0


def test_high_news_event_risk_increases_tail_risk_score() -> None:
    engine = ExpiryDistributionEngine(monte_carlo_simulations=1000, random_seed=21)
    returns = _returns()

    low_risk = engine.estimate(
        spot_price=26000.0,
        historical_returns=returns,
        days_to_expiry=10,
        realized_volatility=0.18,
        news_event_risk={"event_risk_level": "LOW"},
    )
    high_risk = engine.estimate(
        spot_price=26000.0,
        historical_returns=returns,
        days_to_expiry=10,
        realized_volatility=0.18,
        regime_result={"event_risk_score": 0.8},
        news_event_risk={"event_risk_level": "HIGH"},
    )

    assert high_risk.tail_risk_score > low_risk.tail_risk_score


def test_invalid_distribution_inputs_raise_value_error() -> None:
    engine = ExpiryDistributionEngine()

    try:
        engine.estimate(spot_price=0.0, historical_returns=[0.01], days_to_expiry=5)
    except ValueError as exc:
        assert "spot_price" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid spot price")

    try:
        engine.estimate(spot_price=100.0, historical_returns=[0.01], days_to_expiry=0)
    except ValueError as exc:
        assert "days_to_expiry" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid days_to_expiry")
