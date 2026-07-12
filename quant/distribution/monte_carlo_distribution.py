"""Monte Carlo expiry distribution using risk-neutral-style GBM simulation."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np

from quant.distribution.empirical_distribution import clean_returns
from quant.distribution.parametric_distribution import TRADING_DAYS_PER_YEAR, _normalize_volatility


def simulate_gbm_expiry_prices(
    *,
    spot_price: float,
    historical_returns: Sequence[float] | np.ndarray,
    days_to_expiry: int,
    realized_volatility: float | None = None,
    predicted_volatility: float | None = None,
    simulations: int = 5000,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate terminal prices with geometric Brownian motion."""

    if spot_price <= 0:
        raise ValueError("spot_price must be positive")
    horizon = max(1, int(days_to_expiry))
    paths = max(100, int(simulations))
    returns = clean_returns(historical_returns)
    annual_volatility = _normalize_volatility(predicted_volatility)
    if annual_volatility is None:
        annual_volatility = _normalize_volatility(realized_volatility)
    if annual_volatility is None:
        annual_volatility = float(np.std(returns, ddof=1) * sqrt(TRADING_DAYS_PER_YEAR)) if returns.size > 1 else 0.18
    daily_drift = float(np.mean(returns)) if returns.size else 0.0
    daily_volatility = max(annual_volatility / sqrt(TRADING_DAYS_PER_YEAR), 1e-8)

    rng = np.random.default_rng(seed)
    shocks = rng.normal(
        loc=daily_drift - 0.5 * daily_volatility**2,
        scale=daily_volatility,
        size=(paths, horizon),
    )
    terminal_log_returns = shocks.sum(axis=1)
    return np.maximum(spot_price * np.exp(terminal_log_returns), 0.01)
