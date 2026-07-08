"""Black-Scholes Greeks and probability helpers."""

from __future__ import annotations

from math import exp, sqrt

from scipy.stats import norm

from pricing.black_scholes import d1_d2


def delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: str = "call",
) -> float:
    """Return option delta."""

    d1, _ = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    if option_type.lower() in {"call", "ce"}:
        return norm.cdf(d1)
    if option_type.lower() in {"put", "pe"}:
        return norm.cdf(d1) - 1
    raise ValueError("option_type must be call/put or CE/PE")


def gamma(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    """Return option gamma."""

    d1, _ = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return norm.pdf(d1) / (spot * volatility * sqrt(time_to_expiry))


def theta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    per_day: bool = True,
) -> float:
    """Return option theta, per day by default."""

    d1, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    first = -(spot * norm.pdf(d1) * volatility) / (2 * sqrt(time_to_expiry))
    if option_type.lower() in {"call", "ce"}:
        value = first - risk_free_rate * strike * exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    elif option_type.lower() in {"put", "pe"}:
        value = first + risk_free_rate * strike * exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be call/put or CE/PE")
    return value / 365 if per_day else value


def vega(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    """Return option vega per 1 volatility point."""

    d1, _ = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return spot * norm.pdf(d1) * sqrt(time_to_expiry) / 100


def rho(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: str = "call",
) -> float:
    """Return option rho per 1 rate point."""

    _, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    if option_type.lower() in {"call", "ce"}:
        return strike * time_to_expiry * exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
    if option_type.lower() in {"put", "pe"}:
        return -strike * time_to_expiry * exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
    raise ValueError("option_type must be call/put or CE/PE")


def probability_itm(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: str = "call",
) -> float:
    """Return risk-neutral probability of expiring in the money."""

    _, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    if option_type.lower() in {"call", "ce"}:
        return norm.cdf(d2)
    if option_type.lower() in {"put", "pe"}:
        return norm.cdf(-d2)
    raise ValueError("option_type must be call/put or CE/PE")


def probability_otm(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: str = "call",
) -> float:
    """Return risk-neutral probability of expiring out of the money."""

    return 1.0 - probability_itm(
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        option_type=option_type,
    )


def expected_move_from_iv(spot: float, volatility: float, time_to_expiry: float) -> float:
    """Return one-standard-deviation expected move from implied volatility."""

    if spot <= 0 or volatility <= 0 or time_to_expiry <= 0:
        raise ValueError("spot, volatility, and time_to_expiry must be positive")
    return spot * volatility * sqrt(time_to_expiry)
