"""Expiry price distribution engine for NIFTY-style underlyings."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Mapping, Sequence

import numpy as np

from quant.distribution.empirical_distribution import empirical_expiry_prices
from quant.distribution.monte_carlo_distribution import simulate_gbm_expiry_prices
from quant.distribution.parametric_distribution import ParametricDistribution, build_parametric_distribution

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIDENCE_LEVELS = (0.68, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class ExpiryDistributionResult:
    """Estimated expiry distribution with reusable probability helpers."""

    expected_expiry_price: float
    expected_move_points: float
    expected_move_percent: float
    bounds: dict[float, tuple[float, float]]
    tail_risk_score: float
    samples: tuple[float, ...] = field(default_factory=tuple)
    parametric_distribution: ParametricDistribution | None = None
    model_used: str = "combined"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def lower_bound_68(self) -> float:
        """Return lower 68% confidence bound."""

        return self.bounds[0.68][0]

    @property
    def upper_bound_68(self) -> float:
        """Return upper 68% confidence bound."""

        return self.bounds[0.68][1]

    @property
    def lower_bound_80(self) -> float:
        """Return lower 80% confidence bound."""

        return self.bounds[0.80][0]

    @property
    def upper_bound_80(self) -> float:
        """Return upper 80% confidence bound."""

        return self.bounds[0.80][1]

    @property
    def lower_bound_90(self) -> float:
        """Return lower 90% confidence bound."""

        return self.bounds[0.90][0]

    @property
    def upper_bound_90(self) -> float:
        """Return upper 90% confidence bound."""

        return self.bounds[0.90][1]

    @property
    def lower_bound_95(self) -> float:
        """Return lower 95% confidence bound."""

        return self.bounds[0.95][0]

    @property
    def upper_bound_95(self) -> float:
        """Return upper 95% confidence bound."""

        return self.bounds[0.95][1]

    def probability_below_strike(self, strike: float) -> float:
        """Return probability that expiry price finishes below strike."""

        if self.samples:
            sample_array = np.asarray(self.samples, dtype=float)
            return float(np.mean(sample_array < float(strike)))
        if self.parametric_distribution is not None:
            return float(self.parametric_distribution.probability_below(float(strike)))
        return 0.0

    def probability_above_strike(self, strike: float) -> float:
        """Return probability that expiry price finishes above strike."""

        return 1.0 - self.probability_below_strike(strike)

    def probability_between(self, lower: float, upper: float) -> float:
        """Return probability that expiry price finishes between two levels."""

        low = min(float(lower), float(upper))
        high = max(float(lower), float(upper))
        if self.samples:
            sample_array = np.asarray(self.samples, dtype=float)
            return float(np.mean((sample_array >= low) & (sample_array <= high)))
        return max(0.0, self.probability_below_strike(high) - self.probability_below_strike(low))


class ExpiryDistributionEngine:
    """Estimate expiry price distribution using empirical, parametric, and Monte Carlo inputs."""

    def __init__(
        self,
        *,
        confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
        use_parametric: bool = True,
        use_empirical: bool = True,
        use_monte_carlo: bool = True,
        monte_carlo_simulations: int = 5000,
        random_seed: int | None = 42,
    ) -> None:
        """Create an expiry distribution engine."""

        self.confidence_levels = tuple(sorted({round(float(level), 2) for level in confidence_levels}))
        self.use_parametric = use_parametric
        self.use_empirical = use_empirical
        self.use_monte_carlo = use_monte_carlo
        self.monte_carlo_simulations = monte_carlo_simulations
        self.random_seed = random_seed

    def estimate(
        self,
        *,
        spot_price: float,
        historical_returns: Sequence[float] | np.ndarray,
        days_to_expiry: int,
        realized_volatility: float | None = None,
        predicted_volatility: float | None = None,
        regime_result: Any | None = None,
        news_event_risk: Any | None = None,
    ) -> ExpiryDistributionResult:
        """Estimate NIFTY expiry price distribution from available research inputs."""

        self._validate_inputs(spot_price, days_to_expiry)
        parametric = build_parametric_distribution(
            spot_price=spot_price,
            historical_returns=historical_returns,
            days_to_expiry=days_to_expiry,
            realized_volatility=realized_volatility,
            predicted_volatility=predicted_volatility,
        ) if self.use_parametric else None

        sample_sets: list[np.ndarray] = []
        sources: list[str] = []
        if self.use_empirical:
            empirical = empirical_expiry_prices(
                spot_price=spot_price,
                historical_returns=historical_returns,
                days_to_expiry=days_to_expiry,
            )
            if empirical.size:
                sample_sets.append(empirical)
                sources.append("empirical")
        if self.use_monte_carlo:
            simulated = simulate_gbm_expiry_prices(
                spot_price=spot_price,
                historical_returns=historical_returns,
                days_to_expiry=days_to_expiry,
                realized_volatility=realized_volatility,
                predicted_volatility=predicted_volatility,
                simulations=self.monte_carlo_simulations,
                seed=self.random_seed,
            )
            sample_sets.append(simulated)
            sources.append("monte_carlo")

        samples = np.concatenate(sample_sets) if sample_sets else np.array([], dtype=float)
        if samples.size:
            expected_expiry_price = float(np.mean(samples))
            bounds = self._sample_bounds(samples)
        elif parametric is not None:
            expected_expiry_price = float(parametric.expected_price)
            bounds = self._parametric_bounds(parametric)
            sources.append("parametric")
        else:
            raise ValueError("At least one distribution method must be enabled")

        expected_move_points = self._expected_move_points(spot_price, bounds)
        tail_risk_score = self._tail_risk_score(
            samples=samples,
            spot_price=spot_price,
            bounds=bounds,
            regime_result=regime_result,
            news_event_risk=news_event_risk,
        )
        LOGGER.info(
            "Expiry distribution estimated",
            extra={
                "spot_price": spot_price,
                "days_to_expiry": days_to_expiry,
                "sources": sources,
                "tail_risk_score": tail_risk_score,
            },
        )
        return ExpiryDistributionResult(
            expected_expiry_price=expected_expiry_price,
            expected_move_points=expected_move_points,
            expected_move_percent=expected_move_points / spot_price * 100.0,
            bounds=bounds,
            tail_risk_score=tail_risk_score,
            samples=tuple(float(value) for value in samples) if samples.size else (),
            parametric_distribution=parametric,
            model_used="+".join(sources) if sources else "parametric",
            metadata={
                "spot_price": spot_price,
                "days_to_expiry": days_to_expiry,
                "realized_volatility": realized_volatility,
                "predicted_volatility": predicted_volatility,
                "sample_count": int(samples.size),
            },
        )

    def _sample_bounds(self, samples: np.ndarray) -> dict[float, tuple[float, float]]:
        bounds: dict[float, tuple[float, float]] = {}
        for level in self.confidence_levels:
            tail = (1.0 - level) / 2.0
            bounds[level] = (
                float(np.quantile(samples, tail)),
                float(np.quantile(samples, 1.0 - tail)),
            )
        return _ensure_default_bounds(bounds)

    def _parametric_bounds(self, distribution: ParametricDistribution) -> dict[float, tuple[float, float]]:
        bounds: dict[float, tuple[float, float]] = {}
        for level in self.confidence_levels:
            tail = (1.0 - level) / 2.0
            bounds[level] = (
                float(distribution.quantile(tail)),
                float(distribution.quantile(1.0 - tail)),
            )
        return _ensure_default_bounds(bounds)

    @staticmethod
    def _expected_move_points(spot_price: float, bounds: Mapping[float, tuple[float, float]]) -> float:
        lower, upper = bounds[0.68]
        return max(abs(spot_price - lower), abs(upper - spot_price))

    @staticmethod
    def _tail_risk_score(
        *,
        samples: np.ndarray,
        spot_price: float,
        bounds: Mapping[float, tuple[float, float]],
        regime_result: Any | None,
        news_event_risk: Any | None,
    ) -> float:
        lower_95, upper_95 = bounds[0.95]
        if samples.size:
            empirical_tail = float(np.mean((samples < lower_95) | (samples > upper_95)))
        else:
            empirical_tail = 0.05
        band_width_percent = max(0.0, (upper_95 - lower_95) / spot_price)
        regime_penalty = _risk_penalty(_read_attr_or_key(regime_result, "event_risk_score"))
        news_level = str(_read_attr_or_key(news_event_risk, "event_risk_level") or "").upper()
        news_penalty = {"LOW": 0.0, "MEDIUM": 0.1, "HIGH": 0.25, "EXTREME": 0.4}.get(news_level, 0.0)
        return float(min(1.0, empirical_tail + band_width_percent * 0.5 + regime_penalty + news_penalty))

    @staticmethod
    def _validate_inputs(spot_price: float, days_to_expiry: int) -> None:
        if spot_price <= 0:
            raise ValueError("spot_price must be positive")
        if days_to_expiry <= 0:
            raise ValueError("days_to_expiry must be positive")


def _ensure_default_bounds(bounds: dict[float, tuple[float, float]]) -> dict[float, tuple[float, float]]:
    for level in DEFAULT_CONFIDENCE_LEVELS:
        rounded = round(level, 2)
        if rounded not in bounds:
            nearest = min(bounds, key=lambda item: abs(item - rounded))
            bounds[rounded] = bounds[nearest]
    return bounds


def _read_attr_or_key(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _risk_penalty(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(score, 1.0)) * 0.2
