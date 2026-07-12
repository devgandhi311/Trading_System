"""Expected-move model blending realized, IV, empirical, and Monte Carlo estimates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping

import numpy as np

from quant.distribution.empirical_distribution import empirical_expiry_prices
from quant.distribution.monte_carlo_distribution import simulate_gbm_expiry_prices
from quant.distribution.parametric_distribution import TRADING_DAYS_PER_YEAR, _normalize_volatility


@dataclass(frozen=True)
class ExpectedMoveResult:
    """Expected move estimate around spot."""

    expected_move_points: float
    expected_move_percent: float
    lower_price: float
    upper_price: float
    confidence: float
    method_used: str
    explanation: str


class ExpectedMoveModel:
    """Calculate and blend expected moves from volatility and distribution inputs."""

    def calculate_from_realized_vol(
        self,
        *,
        spot_price: float,
        realized_volatility: float,
        days_to_expiry: int,
    ) -> ExpectedMoveResult:
        """Estimate expected move using annualized realized volatility."""

        self._validate_inputs(spot_price, days_to_expiry)
        annual_vol = _normalize_volatility(realized_volatility) or 0.0
        move_points = spot_price * annual_vol * sqrt(days_to_expiry / TRADING_DAYS_PER_YEAR)
        return self._result(
            spot_price=spot_price,
            move_points=move_points,
            confidence=0.55,
            method="realized_vol",
            explanation="Expected move from annualized realized volatility.",
        )

    def calculate_from_iv(
        self,
        *,
        spot_price: float,
        implied_volatility: float,
        days_to_expiry: int,
    ) -> ExpectedMoveResult:
        """Estimate expected move using option implied volatility."""

        self._validate_inputs(spot_price, days_to_expiry)
        annual_iv = _normalize_volatility(implied_volatility) or 0.0
        move_points = spot_price * annual_iv * sqrt(days_to_expiry / TRADING_DAYS_PER_YEAR)
        return self._result(
            spot_price=spot_price,
            move_points=move_points,
            confidence=0.70,
            method="iv",
            explanation="Expected move from annualized implied volatility.",
        )

    def calculate_from_empirical_returns(
        self,
        *,
        spot_price: float,
        historical_returns: Sequence[float] | np.ndarray,
        days_to_expiry: int,
        confidence_level: float = 0.68,
    ) -> ExpectedMoveResult:
        """Estimate expected move from empirical rolling expiry prices."""

        self._validate_inputs(spot_price, days_to_expiry)
        prices = empirical_expiry_prices(
            spot_price=spot_price,
            historical_returns=historical_returns,
            days_to_expiry=days_to_expiry,
        )
        if prices.size == 0:
            return self._result(
                spot_price=spot_price,
                move_points=0.0,
                confidence=0.0,
                method="empirical_unavailable",
                explanation="Not enough historical returns for empirical expected move.",
            )
        tail = (1.0 - confidence_level) / 2.0
        lower = float(np.quantile(prices, tail))
        upper = float(np.quantile(prices, 1.0 - tail))
        move_points = max(abs(spot_price - lower), abs(upper - spot_price))
        confidence = min(0.75, 0.35 + prices.size / 1000.0)
        return self._result(
            spot_price=spot_price,
            move_points=move_points,
            confidence=confidence,
            method="empirical",
            explanation=f"Expected move from {prices.size} empirical rolling expiry observations.",
        )

    def calculate_from_monte_carlo(
        self,
        *,
        spot_price: float,
        historical_returns: Sequence[float] | np.ndarray,
        days_to_expiry: int,
        realized_volatility: float | None = None,
        predicted_volatility: float | None = None,
        simulations: int = 5000,
        seed: int | None = 42,
        confidence_level: float = 0.68,
    ) -> ExpectedMoveResult:
        """Estimate expected move from GBM Monte Carlo terminal prices."""

        self._validate_inputs(spot_price, days_to_expiry)
        prices = simulate_gbm_expiry_prices(
            spot_price=spot_price,
            historical_returns=historical_returns,
            days_to_expiry=days_to_expiry,
            realized_volatility=realized_volatility,
            predicted_volatility=predicted_volatility,
            simulations=simulations,
            seed=seed,
        )
        tail = (1.0 - confidence_level) / 2.0
        lower = float(np.quantile(prices, tail))
        upper = float(np.quantile(prices, 1.0 - tail))
        move_points = max(abs(spot_price - lower), abs(upper - spot_price))
        return self._result(
            spot_price=spot_price,
            move_points=move_points,
            confidence=0.65,
            method="monte_carlo",
            explanation=f"Expected move from {len(prices)} GBM Monte Carlo simulations.",
        )

    def blend_expected_move(
        self,
        *,
        spot_price: float,
        days_to_expiry: int,
        historical_returns: Sequence[float] | np.ndarray = (),
        realized_volatility: float | None = None,
        implied_volatility: float | None = None,
        empirical_move: ExpectedMoveResult | None = None,
        monte_carlo_move: ExpectedMoveResult | None = None,
        regime_result: Any | None = None,
        news_event_risk: Any | None = None,
    ) -> ExpectedMoveResult:
        """Blend expected move sources with regime volatility and news risk multipliers."""

        self._validate_inputs(spot_price, days_to_expiry)
        components: list[tuple[ExpectedMoveResult, float]] = []
        if realized_volatility is not None:
            components.append(
                (
                    self.calculate_from_realized_vol(
                        spot_price=spot_price,
                        realized_volatility=realized_volatility,
                        days_to_expiry=days_to_expiry,
                    ),
                    0.25,
                )
            )
        if implied_volatility is not None:
            components.append(
                (
                    self.calculate_from_iv(
                        spot_price=spot_price,
                        implied_volatility=implied_volatility,
                        days_to_expiry=days_to_expiry,
                    ),
                    0.35,
                )
            )
        if empirical_move is None and _has_values(historical_returns):
            empirical_move = self.calculate_from_empirical_returns(
                spot_price=spot_price,
                historical_returns=historical_returns,
                days_to_expiry=days_to_expiry,
            )
        if empirical_move is not None and empirical_move.confidence > 0:
            components.append((empirical_move, 0.25))
        if monte_carlo_move is None and _has_values(historical_returns):
            monte_carlo_move = self.calculate_from_monte_carlo(
                spot_price=spot_price,
                historical_returns=historical_returns,
                days_to_expiry=days_to_expiry,
                realized_volatility=realized_volatility,
                predicted_volatility=implied_volatility,
            )
        if monte_carlo_move is not None:
            components.append((monte_carlo_move, 0.15))
        if not components:
            raise ValueError("At least one expected move source is required")

        weighted_total = sum(result.expected_move_points * weight for result, weight in components)
        total_weight = sum(weight for _, weight in components)
        move_points = weighted_total / total_weight
        move_points *= self._regime_volatility_multiplier(regime_result)
        move_points *= self._news_risk_multiplier(news_event_risk)
        confidence = sum(result.confidence * weight for result, weight in components) / total_weight
        method = "+".join(result.method_used for result, _ in components)
        return self._result(
            spot_price=spot_price,
            move_points=move_points,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            method=f"blend:{method}",
            explanation=(
                "Blended expected move using available realized volatility, IV, empirical, "
                "and Monte Carlo estimates with regime/news multipliers."
            ),
        )

    @staticmethod
    def _result(
        *,
        spot_price: float,
        move_points: float,
        confidence: float,
        method: str,
        explanation: str,
    ) -> ExpectedMoveResult:
        move = max(float(move_points), 0.0)
        return ExpectedMoveResult(
            expected_move_points=move,
            expected_move_percent=move / spot_price * 100.0 if spot_price > 0 else 0.0,
            lower_price=spot_price - move,
            upper_price=spot_price + move,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            method_used=method,
            explanation=explanation,
        )

    @staticmethod
    def _validate_inputs(spot_price: float, days_to_expiry: int) -> None:
        if spot_price <= 0:
            raise ValueError("spot_price must be positive")
        if days_to_expiry <= 0:
            raise ValueError("days_to_expiry must be positive")

    @staticmethod
    def _regime_volatility_multiplier(regime_result: Any | None) -> float:
        score = _read_float(regime_result, "volatility_score")
        if score is None:
            return 1.0
        return 0.85 + min(max(score, 0.0), 1.0) * 0.35

    @staticmethod
    def _news_risk_multiplier(news_event_risk: Any | None) -> float:
        level = str(_read_value(news_event_risk, "event_risk_level") or "").upper()
        return {"LOW": 1.0, "MEDIUM": 1.08, "HIGH": 1.20, "EXTREME": 1.35}.get(level, 1.0)


def _read_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _read_float(source: Any, name: str) -> float | None:
    value = _read_value(source, name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_values(values: Sequence[float] | np.ndarray) -> bool:
    return np.asarray(values, dtype=float).size > 0
