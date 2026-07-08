"""Backtest report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        """Save an Excel backtest report."""

        path = ensure_parent_dir(self.report_dir / f"{name}.xlsx")
        with pd.ExcelWriter(path) as writer:
            trades.to_excel(writer, sheet_name="trades", index=False)
            equity_curve.to_excel(writer, sheet_name="equity", index=False)
            pd.DataFrame([_stringify(metrics.as_dict())]).to_excel(writer, sheet_name="metrics", index=False)
            if walk_forward is not None:
                walk_forward.to_excel(writer, sheet_name="walk_forward", index=False)
        return path


def _stringify(row: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, dict) else value for key, value in row.items()}
