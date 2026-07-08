"""Daily P&L guardrails."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DailyLossResult:
    """Result for daily loss and profit-lock checks."""

    approved: bool
    reason: str
    warnings: tuple[str, ...]
    daily_pnl: float
    daily_loss_limit: float
    profit_lock_level: float


class DailyLossGuard:
    """Check daily max loss and profit lock."""

    def __init__(
        self,
        *,
        capital: float,
        max_daily_loss_percent: float,
        max_daily_profit_lock_percent: float,
    ) -> None:
        """Create daily P&L guard."""

        self.capital = float(capital)
        self.max_daily_loss_percent = float(max_daily_loss_percent)
        self.max_daily_profit_lock_percent = float(max_daily_profit_lock_percent)

    def check(self, *, daily_pnl: float, peak_daily_pnl: float | None = None) -> DailyLossResult:
        """Validate daily loss and lock profit once peak profit threshold is reached."""

        daily_loss_limit = self.capital * self.max_daily_loss_percent / 100.0
        profit_lock_level = self.capital * self.max_daily_profit_lock_percent / 100.0
        if daily_pnl <= -daily_loss_limit:
            return DailyLossResult(False, "Daily max loss limit hit.", (), daily_pnl, daily_loss_limit, profit_lock_level)
        if peak_daily_pnl is not None and peak_daily_pnl >= profit_lock_level and daily_pnl < peak_daily_pnl * 0.5:
            return DailyLossResult(False, "Daily profit lock triggered.", (), daily_pnl, daily_loss_limit, profit_lock_level)
        warnings: list[str] = []
        if daily_pnl <= -daily_loss_limit * 0.8:
            warnings.append("daily_loss_near_limit")
        if peak_daily_pnl is not None and peak_daily_pnl >= profit_lock_level:
            warnings.append("daily_profit_lock_active")
        return DailyLossResult(True, "Daily P&L guard passed.", tuple(warnings), daily_pnl, daily_loss_limit, profit_lock_level)
