"""Per-trade loss guardrails."""

from __future__ import annotations

from dataclasses import dataclass

from models import StrategyPlan


@dataclass(frozen=True)
class TradeLossResult:
    """Result of per-trade max loss validation."""

    approved: bool
    reason: str
    max_risk_amount: float
    strategy_max_loss: float


class TradeLossGuard:
    """Validate strategy max loss against configured per-trade risk."""

    def __init__(self, *, capital: float, max_loss_per_trade_percent: float) -> None:
        """Create per-trade loss guard."""

        self.capital = float(capital)
        self.max_loss_per_trade_percent = float(max_loss_per_trade_percent)

    def check(self, plan: StrategyPlan) -> TradeLossResult:
        """Return approval for a plan's max loss."""

        max_risk_amount = self.capital * self.max_loss_per_trade_percent / 100.0
        strategy_max_loss = float(plan.max_loss or 0.0)
        if strategy_max_loss <= 0:
            return TradeLossResult(False, "Strategy max loss is not defined.", max_risk_amount, strategy_max_loss)
        if strategy_max_loss > max_risk_amount:
            return TradeLossResult(False, "Per-trade max loss exceeded.", max_risk_amount, strategy_max_loss)
        return TradeLossResult(True, "Per-trade loss guard passed.", max_risk_amount, strategy_max_loss)
