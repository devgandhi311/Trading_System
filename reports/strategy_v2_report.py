"""Excel report writer for Strategy V2 research backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from utils.file_utils import ensure_parent_dir


class StrategyV2ReportWriter:
    """Write the full Strategy V2 backtest workbook."""

    def save(
        self,
        *,
        report_dir: str | Path,
        summary: Mapping[str, Any],
        trades: pd.DataFrame,
        strategy_candidates: pd.DataFrame,
        rejected_candidates: pd.DataFrame,
        expiry_distribution: pd.DataFrame,
        vol_forecasts: pd.DataFrame,
        stress_tests: pd.DataFrame,
        no_trade_days: pd.DataFrame,
        dataset_manifest: Mapping[str, Any],
    ) -> Path:
        """Save final_backtest_V2_report.xlsx with all required sheets."""

        source_type = str(summary.get("data_source", "UNKNOWN")).upper()
        report_path = ensure_parent_dir(Path(report_dir) / f"final_backtest_V2_{source_type}_report.xlsx")
        trades = trades.copy()
        equity_curve = _equity_curve(trades)
        drawdown = _drawdown(equity_curve)
        monthly = _period_returns(trades, "M")
        yearly = _period_returns(trades, "Y")
        cost_drag = _cost_drag(trades)
        gross_net = _gross_vs_net(trades)
        with pd.ExcelWriter(report_path) as writer:
            pd.DataFrame([dict(summary)]).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame([{"field": key, "value": str(value)} for key, value in dataset_manifest.items()]).to_excel(
                writer, sheet_name="Dataset_Manifest", index=False
            )
            _sheet(trades).to_excel(writer, sheet_name="Trades", index=False)
            _sheet(strategy_candidates).to_excel(writer, sheet_name="Strategy Candidates", index=False)
            _sheet(rejected_candidates).to_excel(writer, sheet_name="Rejected Candidates", index=False)
            _sheet(expiry_distribution).to_excel(writer, sheet_name="Expiry Distribution", index=False)
            _sheet(vol_forecasts).to_excel(writer, sheet_name="Vol Forecasts", index=False)
            _attribution(trades, "regime").to_excel(writer, sheet_name="Regime Attribution", index=False)
            _attribution(trades, "news_risk").to_excel(writer, sheet_name="News Risk Attribution", index=False)
            _sheet(stress_tests).to_excel(writer, sheet_name="Stress Tests", index=False)
            _sheet(equity_curve).to_excel(writer, sheet_name="Equity Curve", index=False)
            _sheet(drawdown).to_excel(writer, sheet_name="Drawdown", index=False)
            _sheet(monthly).to_excel(writer, sheet_name="Monthly Returns", index=False)
            _sheet(yearly).to_excel(writer, sheet_name="Yearly Returns", index=False)
            _sheet(cost_drag).to_excel(writer, sheet_name="Cost Drag", index=False)
            _sheet(gross_net).to_excel(writer, sheet_name="Gross vs Net", index=False)
            _sheet(no_trade_days).to_excel(writer, sheet_name="No Trade Days", index=False)
        return report_path


def _sheet(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame({"status": ["no rows"]})
    return dataframe


def _equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "net_pnl" not in trades:
        return pd.DataFrame({"trade_number": [0], "equity": [0.0]})
    equity = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0).cumsum()
    return pd.DataFrame({"trade_number": range(1, len(equity) + 1), "equity": equity})


def _drawdown(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame({"trade_number": [0], "drawdown": [0.0]})
    equity = pd.to_numeric(equity_curve["equity"], errors="coerce").fillna(0.0)
    peak = equity.cummax()
    drawdown = equity - peak
    return pd.DataFrame({"trade_number": equity_curve["trade_number"], "drawdown": drawdown})


def _period_returns(trades: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if trades.empty or "date" not in trades:
        return pd.DataFrame({"period": [], "net_pnl": []})
    data = trades.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["period"] = data["date"].dt.to_period(frequency).astype(str)
    return data.groupby("period", dropna=False)["net_pnl"].sum().reset_index()


def _cost_drag(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"transaction_cost": 0.0, "slippage": 0.0, "total_cost_drag": 0.0}])
    transaction_cost = float(pd.to_numeric(trades.get("transaction_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    slippage = float(pd.to_numeric(trades.get("slippage", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    return pd.DataFrame([{"transaction_cost": transaction_cost, "slippage": slippage, "total_cost_drag": transaction_cost + slippage}])


def _gross_vs_net(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"gross_pnl": 0.0, "net_pnl": 0.0, "decay": 0.0}])
    gross = float(pd.to_numeric(trades.get("gross_pnl", 0.0), errors="coerce").fillna(0.0).sum())
    net = float(pd.to_numeric(trades.get("net_pnl", 0.0), errors="coerce").fillna(0.0).sum())
    decay = (gross - net) / abs(gross) if gross else 0.0
    return pd.DataFrame([{"gross_pnl": gross, "net_pnl": net, "decay": decay}])


def _attribution(trades: pd.DataFrame, column: str) -> pd.DataFrame:
    if trades.empty or column not in trades:
        return pd.DataFrame({column: [], "net_pnl": [], "trades": []})
    return trades.groupby(column, dropna=False).agg(net_pnl=("net_pnl", "sum"), trades=("net_pnl", "count")).reset_index()
