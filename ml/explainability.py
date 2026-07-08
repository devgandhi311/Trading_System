"""Explainability helpers for prediction and strategy recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from utils.math_utils import clamp


@dataclass(frozen=True)
class PredictionExplanation:
    """Human-readable explanation payload for a prediction."""

    top_features: tuple[tuple[str, float], ...]
    model_confidence_explanation: str
    warning_flags: tuple[str, ...] = field(default_factory=tuple)
    strategy_suggestion_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "top_features": list(self.top_features),
            "model_confidence_explanation": self.model_confidence_explanation,
            "warning_flags": list(self.warning_flags),
            "strategy_suggestion_reason": self.strategy_suggestion_reason,
        }


class ExplainabilityEngine:
    """Build compact explanations from features, probabilities, and strategy context."""

    def top_features(
        self,
        features: Mapping[str, Any],
        *,
        limit: int = 8,
        importances: Mapping[str, float] | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Return the strongest available features by importance or absolute value."""

        scored: list[tuple[str, float]] = []
        if importances:
            for name, score in importances.items():
                if name in features:
                    scored.append((name, float(score)))
        else:
            for name, value in features.items():
                try:
                    numeric_value = abs(float(value))
                except (TypeError, ValueError):
                    continue
                if numeric_value > 0:
                    scored.append((name, numeric_value))
        scored.sort(key=lambda item: item[1], reverse=True)
        return tuple(scored[:limit])

    def confidence_text(
        self,
        *,
        confidence: float,
        bullish_probability: float,
        bearish_probability: float,
        sideways_probability: float,
    ) -> str:
        """Explain model confidence in plain language."""

        confidence = clamp(confidence, 0.0, 1.0)
        leading = max(
            (
                ("bullish", bullish_probability),
                ("bearish", bearish_probability),
                ("sideways", sideways_probability),
            ),
            key=lambda item: item[1],
        )
        if confidence >= 0.75:
            strength = "high"
        elif confidence >= 0.55:
            strength = "moderate"
        else:
            strength = "low"
        return f"{strength} confidence: {leading[0]} scenario leads with {leading[1]:.1%} probability."

    def warning_flags(
        self,
        features: Mapping[str, Any],
        *,
        trade_allowed: bool,
        breakout_failure_probability: float,
        liquidity_score: float | None = None,
    ) -> tuple[str, ...]:
        """Return warning flags that should reduce confidence or block trading."""

        warnings: list[str] = []
        if not trade_allowed:
            warnings.append("trade_not_allowed_by_regime_or_config")
        if breakout_failure_probability >= 0.6:
            warnings.append("high_breakout_failure_probability")
        if liquidity_score is not None and liquidity_score < 0.35:
            warnings.append("low_liquidity_score")
        spread = _as_float(features.get("bid_ask_spread_percent"))
        if spread is not None and spread > 1.0:
            warnings.append("wide_bid_ask_spread")
        slippage = _as_float(features.get("slippage_estimate"))
        if slippage is not None and slippage > 0.75:
            warnings.append("high_estimated_slippage")
        if bool(features.get("abnormal_candle")):
            warnings.append("abnormal_latest_candle")
        return tuple(warnings)

    def explain_strategy(
        self,
        strategy_type: str | None,
        *,
        regime_name: str,
        confidence: float,
        expected_move_percent: float,
        iv_percentile: float | None,
    ) -> str:
        """Explain why a strategy type was suggested."""

        if not strategy_type:
            return "No strategy suggested because confidence, regime, or risk filters did not pass."
        iv_text = "unknown IV percentile" if iv_percentile is None else f"IV percentile {iv_percentile:.1f}"
        return (
            f"{strategy_type} is suggested for {regime_name} with "
            f"{confidence:.1%} confidence, {expected_move_percent:.2f}% expected move, and {iv_text}."
        )

    def build(
        self,
        features: Mapping[str, Any],
        *,
        confidence: float,
        bullish_probability: float,
        bearish_probability: float,
        sideways_probability: float,
        breakout_failure_probability: float,
        trade_allowed: bool,
        strategy_type: str | None,
        regime_name: str,
        expected_move_percent: float,
        iv_percentile: float | None = None,
        importances: Mapping[str, float] | None = None,
    ) -> PredictionExplanation:
        """Build the full explanation object."""

        liquidity_score = _as_float(features.get("liquidity_score"))
        return PredictionExplanation(
            top_features=self.top_features(features, importances=importances),
            model_confidence_explanation=self.confidence_text(
                confidence=confidence,
                bullish_probability=bullish_probability,
                bearish_probability=bearish_probability,
                sideways_probability=sideways_probability,
            ),
            warning_flags=self.warning_flags(
                features,
                trade_allowed=trade_allowed,
                breakout_failure_probability=breakout_failure_probability,
                liquidity_score=liquidity_score,
            ),
            strategy_suggestion_reason=self.explain_strategy(
                strategy_type,
                regime_name=regime_name,
                confidence=confidence,
                expected_move_percent=expected_move_percent,
                iv_percentile=iv_percentile,
            ),
        )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
