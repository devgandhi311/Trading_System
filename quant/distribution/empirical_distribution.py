"""Empirical expiry distribution from historical daily returns."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


def clean_returns(returns: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return finite decimal returns suitable for distribution estimation."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    # Treat obvious percent inputs as percentages while keeping decimal returns unchanged.
    if np.nanmedian(np.abs(values)) > 1.0:
        values = values / 100.0
    return values


def horizon_returns(returns: Sequence[float] | np.ndarray, days_to_expiry: int) -> np.ndarray:
    """Compound historical returns into rolling horizon returns."""

    cleaned = clean_returns(returns)
    horizon = max(1, int(days_to_expiry))
    if cleaned.size < horizon:
        return np.array([], dtype=float)
    compounded = []
    for index in range(0, cleaned.size - horizon + 1):
        window = cleaned[index : index + horizon]
        compounded.append(float(np.prod(1.0 + window) - 1.0))
    return np.asarray(compounded, dtype=float)


def empirical_expiry_prices(
    *,
    spot_price: float,
    historical_returns: Sequence[float] | np.ndarray,
    days_to_expiry: int,
) -> np.ndarray:
    """Map rolling horizon returns into possible expiry prices."""

    if spot_price <= 0:
        raise ValueError("spot_price must be positive")
    returns = horizon_returns(historical_returns, days_to_expiry)
    if returns.size == 0:
        LOGGER.debug("Not enough returns for empirical expiry distribution")
        return np.array([], dtype=float)
    return np.maximum(spot_price * (1.0 + returns), 0.01)
