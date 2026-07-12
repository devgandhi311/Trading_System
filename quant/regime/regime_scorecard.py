"""Advanced optional regime scorecard combining multiple research signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from models import RegimeResult
from quant.regime.regime_transition import transition_risk


@dataclass(frozen=True)
class AdvancedRegimeResult:
    """Combined regime result from detector, HMM, transitions, volatility, news, and liquidity."""

    current_regime: str
    regime_probability: float
    next_regime_probability: dict[str | int, float]
    transition_risk: float
    trade_allowed: bool
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RegimeScorecard:
    """Build an optional advanced regime layer without replacing the existing detector."""

    def score(
        self,
        *,
        detector_result: RegimeResult | str,
        hmm_regime_probability: Mapping[str | int, float] | None = None,
        next_regime_probability: Mapping[str | int, float] | None = None,
        volatility_forecast: Any | None = None,
        news_risk: Any | None = None,
        liquidity_score: float | None = None,
    ) -> AdvancedRegimeResult:
        """Combine regime context into an advanced optional result."""

        current = _regime_name(detector_result)
        regime_probability = _selected_probability(hmm_regime_probability, current)
        next_prob = dict(next_regime_probability or {})
        risk = transition_risk(next_prob, current)
        news_level = str(_read_value(news_risk, "event_risk_level") or "LOW").upper()
        vol_state = str(_read_value(volatility_forecast, "volatility_state") or "unknown").lower()
        liquidity = 0.5 if liquidity_score is None else float(liquidity_score)
        trade_allowed = bool(getattr(detector_result, "allowed_to_trade", True))
        reasons: list[str] = []
        if news_level in {"HIGH", "EXTREME"}:
            trade_allowed = False
            reasons.append(f"news risk {news_level}")
        if liquidity < 0.35:
            trade_allowed = False
            reasons.append("low liquidity")
        if risk > 0.70:
            trade_allowed = False
            reasons.append("high transition risk")
        if vol_state == "high":
            reasons.append("high volatility forecast")
        if not reasons:
            reasons.append("advanced regime checks acceptable")
        return AdvancedRegimeResult(
            current_regime=current,
            regime_probability=regime_probability,
            next_regime_probability=next_prob,
            transition_risk=risk,
            trade_allowed=trade_allowed,
            explanation=", ".join(reasons),
            metadata={
                "news_event_risk_level": news_level,
                "volatility_state": vol_state,
                "liquidity_score": liquidity,
                "base_trade_allowed": bool(getattr(detector_result, "allowed_to_trade", True)),
            },
        )


def _regime_name(detector_result: RegimeResult | str) -> str:
    if isinstance(detector_result, str):
        return detector_result
    return detector_result.regime_name or detector_result.regime.value


def _selected_probability(probabilities: Mapping[str | int, float] | None, current: str) -> float:
    if not probabilities:
        return 0.0
    if current in probabilities:
        return float(probabilities[current])
    try:
        return float(max(probabilities.values()))
    except ValueError:
        return 0.0


def _read_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)
