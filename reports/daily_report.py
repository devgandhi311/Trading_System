"""Daily trading report aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from reports.trade_report import TradeReportBuilder


@dataclass(frozen=True)
class DailyReportSummary:
    """Daily report summary payload."""

    report_date: date
    total_trades: int
    winning_trades: int
    losing_trades: int
    net_pnl: float
    max_drawdown: float
    daily_loss_usage: float
    best_trade: str
    worst_trade: str
    regime_wise_performance: dict[str, float] = field(default_factory=dict)
    strategy_wise_performance: dict[str, float] = field(default_factory=dict)
    mistakes: tuple[str, ...] = ()
    next_day_improvement_notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a report-friendly dictionary."""

        return {
            "report_date": self.report_date.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "net_pnl": self.net_pnl,
            "max_drawdown": self.max_drawdown,
            "daily_loss_usage": self.daily_loss_usage,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "regime_wise_performance": str(self.regime_wise_performance),
            "strategy_wise_performance": str(self.strategy_wise_performance),
            "mistakes": "; ".join(self.mistakes),
            "next_day_improvement_notes": "; ".join(self.next_day_improvement_notes),
        }


class DailyReportBuilder:
    """Aggregate trade rows into a daily report."""

    def __init__(self, *, capital: float = 1_000_000.0, max_daily_loss_percent: float = 2.0) -> None:
        """Create daily report builder."""

        self.capital = float(capital)
        self.max_daily_loss_percent = float(max_daily_loss_percent)
        self.trade_builder = TradeReportBuilder()

    def build(
        self,
        trades: list[Mapping[str, Any] | Any],
        *,
        report_date: date,
    ) -> DailyReportSummary:
        """Build daily summary from trades."""

        rows = [self.trade_builder.build_row(trade) for trade in trades]
        pnls = [_float(row.get("net_pnl")) for row in rows]
        total = len(rows)
        winners = [value for value in pnls if value > 0]
        losers = [value for value in pnls if value < 0]
        best_index = max(range(total), key=lambda index: pnls[index]) if total else None
        worst_index = min(range(total), key=lambda index: pnls[index]) if total else None
        net_pnl = sum(pnls)
        loss_limit = self.capital * self.max_daily_loss_percent / 100.0
        return DailyReportSummary(
            report_date=report_date,
            total_trades=total,
            winning_trades=len(winners),
            losing_trades=len(losers),
            net_pnl=net_pnl,
            max_drawdown=self._max_intraday_drawdown(pnls),
            daily_loss_usage=max(0.0, -net_pnl / loss_limit) if loss_limit > 0 else 0.0,
            best_trade=str(rows[best_index]["trade_id"]) if best_index is not None else "",
            worst_trade=str(rows[worst_index]["trade_id"]) if worst_index is not None else "",
            regime_wise_performance=_group(rows, "regime_at_entry"),
            strategy_wise_performance=_group(rows, "strategy"),
            mistakes=tuple(str(row["trade_id"]) for row in rows if _truthy(row.get("mistake_flag"))),
            next_day_improvement_notes=tuple(
                str(row["improvement_note"]) for row in rows if str(row.get("improvement_note") or "").strip()
            ),
        )

    @staticmethod
    def _max_intraday_drawdown(pnls: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity - peak)
        return max_drawdown


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for row in rows:
        group = str(row.get(key) or "unknown")
        grouped[group] = grouped.get(group, 0.0) + _float(row.get("net_pnl"))
    return grouped


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
