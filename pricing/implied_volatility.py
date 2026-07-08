"""Implied volatility solving and IV statistics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.optimize import brentq

from pricing.black_scholes import call_price, put_price


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    *,
    option_type: str = "call",
    lower_bound: float = 1e-6,
    upper_bound: float = 5.0,
) -> float:
    """Solve implied volatility with Brent's method."""

    if market_price <= 0:
        raise ValueError("market_price must be positive")

    def objective(volatility: float) -> float:
        if option_type.lower() in {"call", "ce"}:
            return call_price(spot, strike, time_to_expiry, risk_free_rate, volatility) - market_price
        if option_type.lower() in {"put", "pe"}:
            return put_price(spot, strike, time_to_expiry, risk_free_rate, volatility) - market_price
        raise ValueError("option_type must be call/put or CE/PE")

    try:
        return float(brentq(objective, lower_bound, upper_bound, maxiter=100, xtol=1e-8))
    except ValueError as exc:
        raise ValueError("Could not solve implied volatility within bounds") from exc


def iv_rank(current_iv: float, historical_ivs: Iterable[float]) -> float:
    """Return IV rank as 0..1 based on historical min/max."""

    values = _clean_values(historical_ivs)
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        return 0.0
    return float(np.clip((current_iv - minimum) / (maximum - minimum), 0.0, 1.0))


def iv_percentile(current_iv: float, historical_ivs: Iterable[float]) -> float:
    """Return share of historical IV values less than or equal to current IV."""

    values = _clean_values(historical_ivs)
    return float((values <= current_iv).mean())


def _clean_values(values: Iterable[float]) -> np.ndarray:
    array = np.array(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("historical_ivs must contain at least one finite value")
    return array
