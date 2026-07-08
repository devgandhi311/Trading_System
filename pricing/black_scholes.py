"""Black-Scholes option pricing helpers."""

from __future__ import annotations

from math import exp, log, sqrt

from scipy.stats import norm


def call_price(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    """Return Black-Scholes European call price."""

    d1, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return spot * norm.cdf(d1) - strike * exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)


def put_price(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    """Return Black-Scholes European put price."""

    d1, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return strike * exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> tuple[float, float]:
    """Return Black-Scholes d1 and d2."""

    _validate_inputs(spot, strike, time_to_expiry, volatility)
    d1 = (
        log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt(time_to_expiry))
    d2 = d1 - volatility * sqrt(time_to_expiry)
    return d1, d2


def _validate_inputs(spot: float, strike: float, time_to_expiry: float, volatility: float) -> None:
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")
