"""Dynamic hedge recommendation engine.

This module never places orders. It only returns hedge recommendations for
research, monitoring, and manual review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HedgeRecommendation:
    """Recommendation for reducing position risk."""

    hedge_required: bool
    hedge_reason: str
    hedge_legs: tuple[dict[str, Any], ...]
    urgency: str
    estimated_cost: float
    risk_reduction: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicHedgeEngine:
    """Recommend hedges from Greeks, spot/vol changes, regime, news, and tail loss."""

    delta_threshold: float = 0.30
    gamma_threshold: float = 0.08
    near_expiry_days: int = 3
    volatility_move_threshold: float = 0.10
    tail_loss_increase_threshold: float = 0.20

    def recommend(
        self,
        *,
        current_positions: Sequence[Mapping[str, Any]],
        greeks: Mapping[str, Any],
        spot_move: float = 0.0,
        volatility_move: float = 0.0,
        regime_change: str | None = None,
        news_risk: Any | None = None,
        time_to_expiry: int = 7,
        tail_loss_change: float | None = None,
    ) -> HedgeRecommendation:
        """Return a hedge recommendation without placing orders."""

        reasons: list[str] = []
        legs: list[dict[str, Any]] = []
        urgency = "LOW"
        delta = abs(_float(greeks.get("delta")))
        gamma = abs(_float(greeks.get("gamma")))
        news_level = str(_read_value(news_risk, "event_risk_level") or "LOW").upper()

        if delta > self.delta_threshold:
            reasons.append("delta exceeds threshold")
            legs.append(_delta_hedge_leg(current_positions, greeks))
            urgency = "MEDIUM"
        if gamma > self.gamma_threshold and time_to_expiry <= self.near_expiry_days:
            reasons.append("gamma risk high near expiry")
            legs.append({"action": "BUY_WING", "reason": "Reduce gamma/tail risk near expiry", "quantity": 1})
            urgency = "HIGH"
        if news_level == "EXTREME":
            reasons.append("EXTREME news risk requires risk reduction")
            legs.append({"action": "REDUCE_POSITION", "reason": "Extreme news risk", "quantity_percent": 50})
            urgency = "CRITICAL"
        if abs(volatility_move) >= self.volatility_move_threshold:
            reasons.append("volatility move exceeds threshold")
            legs.append({"action": "BUY_OPTION_HEDGE", "reason": "Volatility shock hedge", "quantity": 1})
            urgency = _max_urgency(urgency, "MEDIUM")
        if regime_change:
            reasons.append(f"regime changed to {regime_change}")
            urgency = _max_urgency(urgency, "MEDIUM")
        if tail_loss_change is not None and tail_loss_change >= self.tail_loss_increase_threshold:
            reasons.append("short premium tail loss increased")
            legs.append({"action": "BUY_WING_OR_REDUCE_QTY", "reason": "Tail loss expansion", "quantity": 1})
            urgency = _max_urgency(urgency, "HIGH")

        hedge_required = bool(reasons)
        estimated_cost = _estimated_cost(legs, current_positions)
        risk_reduction = _risk_reduction_score(urgency, len(legs), hedge_required)
        return HedgeRecommendation(
            hedge_required=hedge_required,
            hedge_reason="; ".join(reasons) if reasons else "No hedge required.",
            hedge_legs=tuple(legs),
            urgency=urgency if hedge_required else "NONE",
            estimated_cost=estimated_cost,
            risk_reduction=risk_reduction,
            explanation=(
                "Hedge recommended from delta/gamma/news/tail-risk rules."
                if hedge_required
                else "Greeks, news, volatility, and tail-risk inputs are within thresholds."
            ),
            metadata={
                "delta": delta,
                "gamma": gamma,
                "spot_move": spot_move,
                "volatility_move": volatility_move,
                "news_risk": news_level,
                "time_to_expiry": time_to_expiry,
                "tail_loss_change": tail_loss_change,
            },
        )


def _delta_hedge_leg(current_positions: Sequence[Mapping[str, Any]], greeks: Mapping[str, Any]) -> dict[str, Any]:
    delta = _float(greeks.get("delta"))
    symbol = str(current_positions[0].get("underlying", current_positions[0].get("symbol", "UNDERLYING"))) if current_positions else "UNDERLYING"
    return {
        "action": "SELL_UNDERLYING" if delta > 0 else "BUY_UNDERLYING",
        "symbol": symbol,
        "quantity": max(1, int(abs(delta) * 100)),
        "reason": "Delta hedge",
    }


def _estimated_cost(legs: Sequence[Mapping[str, Any]], positions: Sequence[Mapping[str, Any]]) -> float:
    del positions
    total = 0.0
    for leg in legs:
        total += float(leg.get("estimated_price", leg.get("premium", 0.0)) or 0.0) * float(leg.get("quantity", 1) or 1)
    return total


def _risk_reduction_score(urgency: str, leg_count: int, hedge_required: bool) -> float:
    if not hedge_required:
        return 0.0
    base = {"LOW": 0.20, "MEDIUM": 0.35, "HIGH": 0.55, "CRITICAL": 0.75}.get(urgency, 0.20)
    return min(1.0, base + leg_count * 0.05)


def _max_urgency(left: str, right: str) -> str:
    order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _read_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
