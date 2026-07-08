"""Position sizing helpers for risk-approved strategy plans."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from models import StrategyPlan


@dataclass(frozen=True)
class PositionSizingResult:
    """Quantity allowed after per-trade risk sizing."""

    allowed_quantity: int
    max_risk_amount: float
    plan_risk: float
    approved: bool
    reason: str


class PositionSizer:
    """Calculate allowed quantity from capital and strategy max loss."""

    def __init__(self, *, capital: float, max_loss_per_trade_percent: float) -> None:
        """Create a sizer using account capital and per-trade risk percent."""

        self.capital = float(capital)
        self.max_loss_per_trade_percent = float(max_loss_per_trade_percent)

    def size(self, plan: StrategyPlan, *, event_risk_level: str | None = None, position_size_multiplier: float | None = None) -> PositionSizingResult:
        """Return allowed quantity and max risk amount for a plan."""

        max_risk_amount = self.capital * self.max_loss_per_trade_percent / 100.0
        plan_risk = float(plan.max_loss or 0.0)
        if plan_risk <= 0:
            return PositionSizingResult(0, max_risk_amount, plan_risk, False, "Strategy max loss is not defined.")
        if plan_risk > max_risk_amount:
            return PositionSizingResult(0, max_risk_amount, plan_risk, False, "Per-trade max loss exceeded.")
        leg_quantity = min((leg.quantity for leg in plan.legs), default=0)
        if leg_quantity <= 0:
            return PositionSizingResult(0, max_risk_amount, plan_risk, False, "Strategy has no executable legs.")
        risk_per_unit = plan_risk / leg_quantity
        allowed_quantity = leg_quantity if risk_per_unit <= 0 else min(leg_quantity, floor(max_risk_amount / risk_per_unit))
        multiplier = position_size_multiplier
        if multiplier is None:
            multiplier = float(plan.metadata.get("position_size_multiplier", 1.0))
        if (event_risk_level or str(plan.metadata.get("event_risk_level", ""))).upper() == "MEDIUM":
            multiplier = min(float(multiplier), 0.5)
        allowed_quantity = floor(allowed_quantity * max(0.0, min(1.0, float(multiplier))))
        return PositionSizingResult(
            allowed_quantity=max(0, allowed_quantity),
            max_risk_amount=max_risk_amount,
            plan_risk=plan_risk,
            approved=allowed_quantity > 0,
            reason="Position size approved." if allowed_quantity > 0 else "Allowed quantity is zero.",
        )
