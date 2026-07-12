"""Portfolio risk optimizer for research candidate sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PositionSizeRecommendation:
    """Position sizing recommendation for a research candidate."""

    quantity: int
    risk_amount: float
    capital_at_risk_percent: float
    var_contribution: float
    cvar_contribution: float
    reason: str


@dataclass
class PortfolioRiskOptimizer:
    """Size candidates using fixed fractional, volatility, VaR, and Kelly-style limits."""

    lot_size: int = 1
    max_var_percent: float = 1.0
    max_cvar_percent: float = 1.5

    def recommend_size(
        self,
        *,
        candidate_strategy: Any,
        capital: float,
        max_risk_per_trade: float,
        max_daily_risk: float,
        current_open_positions: Sequence[Mapping[str, Any]] = (),
        var: float = 0.0,
        cvar: float = 0.0,
        correlation_estimate: float = 0.0,
        news_risk: Any | None = None,
        method: str = "fixed_fractional",
        volatility: float | None = None,
    ) -> PositionSizeRecommendation:
        """Return position size recommendation or zero quantity with rejection reason."""

        news_level = str(_read_value(news_risk, "event_risk_level") or "LOW").upper()
        if news_level in {"HIGH", "EXTREME"}:
            return _reject("News risk too high.")
        max_loss = float(getattr(candidate_strategy, "max_loss", 0.0) or 0.0)
        if max_loss <= 0:
            return _reject("Candidate max loss is unavailable.")
        risk_budget = min(float(max_risk_per_trade), float(max_daily_risk), float(capital) * 0.02)
        open_risk = sum(float(position.get("risk_amount", position.get("max_loss", 0.0)) or 0.0) for position in current_open_positions)
        available_daily_risk = max(0.0, float(max_daily_risk) - open_risk)
        risk_budget = min(risk_budget, available_daily_risk)
        if method == "volatility_adjusted" and volatility:
            risk_budget *= max(0.25, min(1.0, 0.20 / abs(float(volatility))))
        if method == "fractional_kelly":
            edge = max(float(getattr(candidate_strategy, "expected_value", 0.0) or 0.0) / max(max_loss, 1.0), 0.0)
            risk_budget *= min(0.5, edge)
        quantity = int(risk_budget // max_loss)
        quantity = (quantity // self.lot_size) * self.lot_size if self.lot_size > 1 else quantity
        if quantity < 1:
            return _reject("quantity < 1 lot")
        risk_amount = quantity * max_loss
        var_contribution = abs(float(var)) * (1.0 + max(float(correlation_estimate), 0.0)) + risk_amount
        cvar_contribution = abs(float(cvar)) * (1.0 + max(float(correlation_estimate), 0.0)) + risk_amount
        if var_contribution > capital * self.max_var_percent / 100.0:
            return _reject("VaR limit exceeded.")
        if cvar_contribution > capital * self.max_cvar_percent / 100.0:
            return _reject("CVaR limit exceeded.")
        return PositionSizeRecommendation(
            quantity=quantity,
            risk_amount=risk_amount,
            capital_at_risk_percent=risk_amount / capital * 100.0 if capital > 0 else 0.0,
            var_contribution=var_contribution,
            cvar_contribution=cvar_contribution,
            reason=f"Sized with {method}.",
        )


def _reject(reason: str) -> PositionSizeRecommendation:
    return PositionSizeRecommendation(0, 0.0, 0.0, 0.0, 0.0, reason)


def _read_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)
