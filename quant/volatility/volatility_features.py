"""Volatility feature helpers used by quant forecasters."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def clean_returns(returns: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return finite decimal returns."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size and np.nanmedian(np.abs(values)) > 1.0:
        values = values / 100.0
    return values


def realized_volatility(
    returns: Sequence[float] | np.ndarray,
    *,
    window: int = 20,
    annualize: bool = True,
) -> float:
    """Calculate rolling realized volatility from recent returns."""

    cleaned = clean_returns(returns)
    if cleaned.size < 2:
        return 0.0
    lookback = cleaned[-max(2, int(window)) :]
    volatility = float(np.std(lookback, ddof=1))
    return volatility * sqrt(TRADING_DAYS_PER_YEAR) if annualize else volatility


def volatility_state(volatility: float, history: Sequence[float] | np.ndarray | None = None) -> str:
    """Classify volatility relative to history or broad defaults."""

    vol = abs(float(volatility))
    if history is not None:
        cleaned = clean_returns(history)
        if cleaned.size > 20:
            annualized = np.asarray(
                [
                    np.std(cleaned[max(0, index - 20) : index], ddof=1) * sqrt(TRADING_DAYS_PER_YEAR)
                    for index in range(20, cleaned.size + 1)
                ],
                dtype=float,
            )
            percentile = float(np.mean(annualized <= vol))
            if percentile >= 0.7:
                return "high"
            if percentile <= 0.3:
                return "low"
            return "normal"
    if vol >= 0.25:
        return "high"
    if vol <= 0.12:
        return "low"
    return "normal"


def horizon_volatility(annualized_volatility: float, days: int) -> float:
    """Convert annualized volatility to horizon volatility."""

    return abs(float(annualized_volatility)) * sqrt(max(1, int(days)) / TRADING_DAYS_PER_YEAR)
