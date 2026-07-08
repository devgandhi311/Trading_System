"""Backtest performance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceSummary:
    """Core backtest metrics."""

    total_return: float
    cagr: float
    max_drawdown: float
    win_rate: float
    average_profit: float
    average_loss: float
    expectancy: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_losing_streak: int
    daily_var: float
    daily_cvar: float
    strategy_wise_pnl: dict[str, float] = field(default_factory=dict)
    regime_wise_pnl: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dictionary for reports."""

        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "average_profit": self.average_profit,
            "average_loss": self.average_loss,
            "expectancy": self.expectancy,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_losing_streak": self.max_losing_streak,
            "daily_var": self.daily_var,
            "daily_cvar": self.daily_cvar,
            "strategy_wise_pnl": self.strategy_wise_pnl,
            "regime_wise_pnl": self.regime_wise_pnl,
        }


class PerformanceMetrics:
    """Calculate performance metrics from trades and equity curve."""

    def calculate(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        *,
        initial_capital: float,
    ) -> PerformanceSummary:
        """Calculate a complete performance summary."""

        pnl = pd.to_numeric(trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        equity = pd.to_numeric(equity_curve.get("equity", pd.Series([initial_capital])), errors="coerce").dropna()
        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] - initial_capital) / initial_capital if not equity.empty else 0.0
        days = max(1, len(equity))
        cagr = (1.0 + total_return) ** (252.0 / days) - 1.0 if total_return > -1 else -1.0
        drawdown = self.drawdown(equity)
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]
        win_rate = float(len(winners) / len(pnl)) if len(pnl) else 0.0
        average_profit = float(winners.mean()) if len(winners) else 0.0
        average_loss = float(losers.mean()) if len(losers) else 0.0
        expectancy = win_rate * average_profit + (1.0 - win_rate) * average_loss
        gross_profit = float(winners.sum()) if len(winners) else 0.0
        gross_loss = abs(float(losers.sum())) if len(losers) else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf") if gross_profit else 0.0
        sharpe = _annualized_ratio(returns)
        downside = returns[returns < 0]
        sortino = _annualized_ratio(returns, denominator=downside.std(ddof=0))
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0
        daily_var = float(returns.quantile(0.05)) if len(returns) else 0.0
        tail = returns[returns <= daily_var]
        daily_cvar = float(tail.mean()) if len(tail) else daily_var
        return PerformanceSummary(
            total_return=float(total_return),
            cagr=float(cagr),
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            average_profit=average_profit,
            average_loss=average_loss,
            expectancy=float(expectancy),
            profit_factor=float(profit_factor),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            max_losing_streak=max_losing_streak(pnl.tolist()),
            daily_var=daily_var,
            daily_cvar=daily_cvar,
            strategy_wise_pnl=_group_pnl(trades, "strategy_type"),
            regime_wise_pnl=_group_pnl(trades, "regime"),
        )

    @staticmethod
    def drawdown(equity: pd.Series) -> pd.Series:
        """Return drawdown series."""

        peak = equity.cummax()
        return equity / peak - 1.0


def max_losing_streak(values: list[float]) -> int:
    """Return max consecutive losing trade count."""

    current = 0
    maximum = 0
    for value in values:
        if value < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _annualized_ratio(returns: pd.Series, denominator: float | None = None) -> float:
    if returns.empty:
        return 0.0
    std = float(denominator if denominator is not None else returns.std(ddof=0))
    if std <= 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(252.0))


def _group_pnl(trades: pd.DataFrame, column: str) -> dict[str, float]:
    if trades.empty or column not in trades.columns or "net_pnl" not in trades.columns:
        return {}
    grouped = trades.groupby(column)["net_pnl"].sum()
    return {str(index): float(value) for index, value in grouped.items()}
