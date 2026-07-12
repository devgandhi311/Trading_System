"""Parametric normal and lognormal expiry distribution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, log, sqrt
from statistics import NormalDist

import numpy as np

from quant.distribution.empirical_distribution import clean_returns

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ParametricDistribution:
    """Lognormal terminal-price approximation with normal CDF helpers."""

    spot_price: float
    days_to_expiry: int
    daily_drift: float
    annualized_volatility: float

    @property
    def years_to_expiry(self) -> float:
        """Return expiry horizon in trading years."""

        return max(1, self.days_to_expiry) / TRADING_DAYS_PER_YEAR

    @property
    def sigma_t(self) -> float:
        """Return horizon volatility."""

        return max(self.annualized_volatility, 1e-8) * sqrt(self.years_to_expiry)

    @property
    def mu_t(self) -> float:
        """Return horizon log mean."""

        return log(self.spot_price) + (self.daily_drift * max(1, self.days_to_expiry)) - 0.5 * self.sigma_t**2

    @property
    def expected_price(self) -> float:
        """Return expected terminal price."""

        return exp(self.mu_t + 0.5 * self.sigma_t**2)

    def quantile(self, probability: float) -> float:
        """Return terminal price quantile."""

        clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        z_score = NormalDist().inv_cdf(clipped)
        return exp(self.mu_t + self.sigma_t * z_score)

    def probability_below(self, strike: float) -> float:
        """Return probability that terminal price is below strike."""

        if strike <= 0:
            return 0.0
        z_score = (log(strike) - self.mu_t) / self.sigma_t
        return NormalDist().cdf(z_score)


def build_parametric_distribution(
    *,
    spot_price: float,
    historical_returns: Sequence[float] | np.ndarray,
    days_to_expiry: int,
    realized_volatility: float | None = None,
    predicted_volatility: float | None = None,
) -> ParametricDistribution:
    """Create a lognormal approximation from historical drift and available volatility."""

    if spot_price <= 0:
        raise ValueError("spot_price must be positive")
    cleaned = clean_returns(historical_returns)
    daily_drift = float(np.mean(cleaned)) if cleaned.size else 0.0
    annualized_vol = _normalize_volatility(predicted_volatility)
    if annualized_vol is None:
        annualized_vol = _normalize_volatility(realized_volatility)
    if annualized_vol is None:
        annualized_vol = float(np.std(cleaned, ddof=1) * sqrt(TRADING_DAYS_PER_YEAR)) if cleaned.size > 1 else 0.18
    return ParametricDistribution(
        spot_price=spot_price,
        days_to_expiry=max(1, int(days_to_expiry)),
        daily_drift=daily_drift,
        annualized_volatility=max(annualized_vol, 1e-6),
    )


def _normalize_volatility(value: float | None) -> float | None:
    if value is None:
        return None
    volatility = abs(float(value))
    if volatility <= 0:
        return None
    return volatility / 100.0 if volatility > 1.0 else volatility
