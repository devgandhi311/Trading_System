"""Backtest report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import pandas as pd

from backtesting.performance_metrics import PerformanceSummary
from utils.file_utils import ensure_parent_dir


class BacktestReportWriter:
    """Write backtest trades, equity, metrics, and walk-forward reports."""

    def __init__(self, *, report_dir: str | Path = "reports/backtests") -> None:
        """Create report writer."""

        self.report_dir = Path(report_dir)

    def save(
        self,
        *,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        metrics: PerformanceSummary,
        name: str = "backtest_report",
        walk_forward: pd.DataFrame | None = None,
    ) -> Path:
        """Save Excel, CSV, and HTML backtest reports."""

        path = ensure_parent_dir(self.report_dir / f"{name}.xlsx")
        drawdown = _drawdown_frame(equity_curve)
        monthly_pnl = _period_pnl(trades, "ME")
        yearly_pnl = _period_pnl(trades, "YE")
        strategy_pnl = _group_pnl(trades, "strategy_type")
        regime_pnl = _group_pnl(trades, "regime")
        summary = pd.DataFrame([_stringify(metrics.as_dict())])
        with pd.ExcelWriter(path) as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            trades.to_excel(writer, sheet_name="Trades", index=False)
            regime_pnl.to_excel(writer, sheet_name="Regime_PnL", index=False)
            strategy_pnl.to_excel(writer, sheet_name="Strategy_PnL", index=False)
            monthly_pnl.to_excel(writer, sheet_name="Monthly_PnL", index=False)
            yearly_pnl.to_excel(writer, sheet_name="Yearly_PnL", index=False)
            equity_curve.to_excel(writer, sheet_name="Equity_Curve", index=False)
            drawdown.to_excel(writer, sheet_name="Drawdown", index=False)
            if walk_forward is not None:
                walk_forward.to_excel(writer, sheet_name="Walk_Forward", index=False)
        _save_csv_bundle(self.report_dir, name, trades, equity_curve, drawdown, monthly_pnl, yearly_pnl, strategy_pnl, regime_pnl)
        _save_html_report(self.report_dir / f"{name}.html", trades, equity_curve, drawdown, metrics, monthly_pnl, yearly_pnl, strategy_pnl, regime_pnl)
        return path


def _stringify(row: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, dict) else value for key, value in row.items()}


def _drawdown_frame(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return pd.DataFrame(columns=["date", "equity", "drawdown"])
    data = equity_curve.copy()
    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")
    data["drawdown"] = data["equity"] / data["equity"].cummax() - 1.0
    return data


def _period_pnl(trades: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if trades.empty or not {"exit_date", "net_pnl"}.issubset(trades.columns):
        return pd.DataFrame(columns=["period", "gross_pnl", "transaction_cost", "net_pnl", "trades"])
    data = trades.copy()
    data["exit_date"] = pd.to_datetime(data["exit_date"], errors="coerce")
    data["gross_pnl"] = pd.to_numeric(data.get("gross_pnl", 0.0), errors="coerce").fillna(0.0)
    data["transaction_cost"] = pd.to_numeric(data.get("transaction_cost", 0.0), errors="coerce").fillna(0.0)
    data["net_pnl"] = pd.to_numeric(data["net_pnl"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["exit_date"]).set_index("exit_date")
    grouped = data.resample(frequency).agg(
        gross_pnl=("gross_pnl", "sum"),
        transaction_cost=("transaction_cost", "sum"),
        net_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "count"),
    )
    return grouped.reset_index().rename(columns={"exit_date": "period"})


def _group_pnl(trades: pd.DataFrame, column: str) -> pd.DataFrame:
    if trades.empty or column not in trades.columns or "net_pnl" not in trades.columns:
        return pd.DataFrame(columns=[column, "gross_pnl", "transaction_cost", "net_pnl", "trades"])
    data = trades.copy()
    data["gross_pnl"] = pd.to_numeric(data.get("gross_pnl", 0.0), errors="coerce").fillna(0.0)
    data["transaction_cost"] = pd.to_numeric(data.get("transaction_cost", 0.0), errors="coerce").fillna(0.0)
    data["net_pnl"] = pd.to_numeric(data["net_pnl"], errors="coerce").fillna(0.0)
    return data.groupby(column).agg(
        gross_pnl=("gross_pnl", "sum"),
        transaction_cost=("transaction_cost", "sum"),
        net_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "count"),
    ).reset_index()


def _save_csv_bundle(
    report_dir: Path,
    name: str,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    drawdown: pd.DataFrame,
    monthly_pnl: pd.DataFrame,
    yearly_pnl: pd.DataFrame,
    strategy_pnl: pd.DataFrame,
    regime_pnl: pd.DataFrame,
) -> None:
    csv_dir = report_dir / f"{name}_csv"
    ensure_parent_dir(csv_dir / "placeholder.txt")
    trades.to_csv(csv_dir / "trades.csv", index=False)
    equity_curve.to_csv(csv_dir / "equity_curve.csv", index=False)
    drawdown.to_csv(csv_dir / "drawdown.csv", index=False)
    monthly_pnl.to_csv(csv_dir / "monthly_pnl.csv", index=False)
    yearly_pnl.to_csv(csv_dir / "yearly_pnl.csv", index=False)
    strategy_pnl.to_csv(csv_dir / "strategy_pnl.csv", index=False)
    regime_pnl.to_csv(csv_dir / "regime_pnl.csv", index=False)


def _save_html_report(
    path: Path,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    drawdown: pd.DataFrame,
    metrics: PerformanceSummary,
    monthly_pnl: pd.DataFrame,
    yearly_pnl: pd.DataFrame,
    strategy_pnl: pd.DataFrame,
    regime_pnl: pd.DataFrame,
) -> None:
    ensure_parent_dir(path)
    sections = {
        "metrics": pd.DataFrame([_stringify(metrics.as_dict())]),
        "trades": trades.head(200),
        "equity_curve": equity_curve,
        "drawdown_chart_data": drawdown,
        "monthly_pnl": monthly_pnl,
        "yearly_pnl": yearly_pnl,
        "strategy_pnl": strategy_pnl,
        "regime_pnl": regime_pnl,
    }
    html = ["<html><head><title>Backtest Report</title></head><body>", "<h1>Backtest Report</h1>"]
    html.append(f"<script type=\"application/json\" id=\"metrics-json\">{json.dumps(_stringify(metrics.as_dict()))}</script>")
    for title, frame in sections.items():
        html.append(f"<h2>{title.replace('_', ' ').title()}</h2>")
        html.append(frame.to_html(index=False))
    html.append("</body></html>")
    path.write_text("\n".join(html), encoding="utf-8")
