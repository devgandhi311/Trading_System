"""Performance attribution for backtest trades.

Run with:
    python -m quant.attribution.performance_attribution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from utils.file_utils import ensure_parent_dir

REPORT_PATH = Path("reports/risk_reports/performance_attribution.xlsx")


@dataclass(frozen=True)
class PerformanceAttributionResult:
    """Attribution tables for strategy research/backtests."""

    pnl_by_strategy: pd.DataFrame
    pnl_by_regime: pd.DataFrame
    pnl_by_volatility_state: pd.DataFrame
    pnl_by_news_risk: pd.DataFrame
    pnl_by_day_of_week: pd.DataFrame
    pnl_by_days_to_expiry: pd.DataFrame
    cost_drag: float
    slippage_drag: float
    gross_to_net_decay: float
    best_conditions: dict[str, Any]
    worst_conditions: dict[str, Any]


class PerformanceAttribution:
    """Create P&L attribution from backtest trade rows."""

    def analyze(self, trades: Sequence[Mapping[str, Any]] | pd.DataFrame) -> PerformanceAttributionResult:
        """Return attribution tables and gross-to-net diagnostics."""

        data = pd.DataFrame(trades).copy()
        if data.empty:
            data = pd.DataFrame(columns=["strategy", "regime", "net_pnl", "gross_pnl"])
        data = _normalize_columns(data)
        gross_total = float(data["gross_pnl"].sum())
        net_total = float(data["net_pnl"].sum())
        cost_drag = float(data["transaction_cost"].sum())
        slippage_drag = float(data["slippage"].sum())
        decay = (gross_total - net_total) / abs(gross_total) if gross_total else 0.0
        condition = (
            data.groupby(["strategy", "regime", "volatility_state", "news_risk"], dropna=False)["net_pnl"]
            .sum()
            .reset_index()
        )
        best = condition.sort_values("net_pnl", ascending=False).head(1).to_dict("records")
        worst = condition.sort_values("net_pnl", ascending=True).head(1).to_dict("records")
        return PerformanceAttributionResult(
            pnl_by_strategy=_group(data, "strategy"),
            pnl_by_regime=_group(data, "regime"),
            pnl_by_volatility_state=_group(data, "volatility_state"),
            pnl_by_news_risk=_group(data, "news_risk"),
            pnl_by_day_of_week=_group(data, "day_of_week"),
            pnl_by_days_to_expiry=_group(data, "days_to_expiry"),
            cost_drag=cost_drag,
            slippage_drag=slippage_drag,
            gross_to_net_decay=float(decay),
            best_conditions=best[0] if best else {},
            worst_conditions=worst[0] if worst else {},
        )

    def save(self, result: PerformanceAttributionResult, path: str | Path = REPORT_PATH) -> Path:
        """Save attribution workbook."""

        file_path = ensure_parent_dir(path)
        with pd.ExcelWriter(file_path) as writer:
            result.pnl_by_strategy.to_excel(writer, sheet_name="PnL_By_Strategy", index=False)
            result.pnl_by_regime.to_excel(writer, sheet_name="PnL_By_Regime", index=False)
            result.pnl_by_volatility_state.to_excel(writer, sheet_name="PnL_By_Volatility", index=False)
            result.pnl_by_news_risk.to_excel(writer, sheet_name="PnL_By_News", index=False)
            result.pnl_by_day_of_week.to_excel(writer, sheet_name="PnL_By_Day", index=False)
            result.pnl_by_days_to_expiry.to_excel(writer, sheet_name="PnL_By_DTE", index=False)
            pd.DataFrame(
                [
                    {"metric": "cost_drag", "value": result.cost_drag},
                    {"metric": "slippage_drag", "value": result.slippage_drag},
                    {"metric": "gross_to_net_decay", "value": result.gross_to_net_decay},
                    {"metric": "best_conditions", "value": str(result.best_conditions)},
                    {"metric": "worst_conditions", "value": str(result.worst_conditions)},
                ]
            ).to_excel(writer, sheet_name="Summary", index=False)
        return file_path


def main() -> None:
    """CLI entrypoint using latest backtest trades if available, otherwise sample rows."""

    trades = _load_default_trades()
    attribution = PerformanceAttribution()
    path = attribution.save(attribution.analyze(trades))
    print(f"performance_attribution saved: {path}")
    print(f"trades: {len(trades)}")


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    defaults = {
        "strategy": "UNKNOWN",
        "regime": "UNKNOWN",
        "volatility_state": "UNKNOWN",
        "news_risk": "LOW",
        "days_to_expiry": 0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "transaction_cost": 0.0,
        "slippage": 0.0,
    }
    for column, default in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        normalized["day_of_week"] = normalized["date"].dt.day_name().fillna("UNKNOWN")
    elif "entry_time" in normalized.columns:
        normalized["day_of_week"] = pd.to_datetime(normalized["entry_time"], errors="coerce").dt.day_name().fillna("UNKNOWN")
    else:
        normalized["day_of_week"] = "UNKNOWN"
    for column in ("gross_pnl", "net_pnl", "transaction_cost", "slippage"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)
    return normalized


def _group(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data.groupby(column, dropna=False)
        .agg(
            gross_pnl=("gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
            trades=("net_pnl", "count"),
            win_rate=("net_pnl", lambda values: float((values > 0).mean()) if len(values) else 0.0),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
    )


def _load_default_trades() -> pd.DataFrame:
    candidates = sorted(Path("reports/backtests").glob("*trades*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return pd.read_csv(candidates[0])
    return pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "strategy": "SAMPLE",
                "regime": "Sideways",
                "volatility_state": "normal",
                "news_risk": "LOW",
                "days_to_expiry": 7,
                "gross_pnl": 1000.0,
                "net_pnl": 900.0,
                "transaction_cost": 80.0,
                "slippage": 20.0,
            }
        ]
    )


if __name__ == "__main__":
    main()
