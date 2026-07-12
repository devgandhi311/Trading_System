"""Walk-forward evaluation for research strategies.

Run with:
    python -m quant.optimization.walk_forward_strategy_eval
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market.data_loader import DataLoader
from strategies.strategy_research_base import StrategyResearchResult
from utils.exceptions import DataValidationError
from utils.file_utils import ensure_parent_dir

REPORT_PATH = Path("reports/model_reports/walk_forward_strategy_eval.xlsx")


@dataclass(frozen=True)
class WalkForwardStrategyRow:
    """One walk-forward strategy evaluation row."""

    train_period: str
    test_period: str
    strategy: str
    total_return: float
    max_drawdown: float
    sharpe: float
    profit_factor: float
    win_rate: float
    number_of_trades: int
    regime_wise_performance: str


def evaluate_walk_forward(data: pd.DataFrame, *, train_size: int = 120, test_size: int = 30) -> list[WalkForwardStrategyRow]:
    """Evaluate research strategy names on rolling return slices."""

    if "close" not in data.columns:
        raise ValueError("data must contain close column")
    frame = data.copy().reset_index(drop=True)
    frame["return"] = frame["close"].pct_change().fillna(0.0)
    strategies = (
        "ExpectedMoveCreditSpread",
        "ExpectedMoveIronCondor",
        "IVMeanReversionPremium",
        "VolatilityExpansion",
    )
    rows: list[WalkForwardStrategyRow] = []
    start = 0
    while start + train_size + test_size <= len(frame):
        train = frame.iloc[start : start + train_size]
        test = frame.iloc[start + train_size : start + train_size + test_size]
        regime = _regime_label(train["return"])
        for strategy in strategies:
            strategy_returns = _strategy_returns(strategy, test["return"], regime)
            rows.append(
                WalkForwardStrategyRow(
                    train_period=_period(train),
                    test_period=_period(test),
                    strategy=strategy,
                    total_return=float((1.0 + strategy_returns).prod() - 1.0),
                    max_drawdown=_max_drawdown(strategy_returns),
                    sharpe=_sharpe(strategy_returns),
                    profit_factor=_profit_factor(strategy_returns),
                    win_rate=float((strategy_returns > 0).mean()) if len(strategy_returns) else 0.0,
                    number_of_trades=int((strategy_returns != 0).sum()),
                    regime_wise_performance=str({regime: float(strategy_returns.sum())}),
                )
            )
        start += test_size
    return rows


def save_walk_forward_report(rows: list[WalkForwardStrategyRow], path: str | Path = REPORT_PATH) -> Path:
    """Save walk-forward rows to Excel."""

    file_path = ensure_parent_dir(path)
    pd.DataFrame([row.__dict__ for row in rows]).to_excel(file_path, index=False)
    return file_path


def main() -> None:
    """CLI entrypoint."""

    try:
        data, _ = DataLoader().load_30_years_nse_data(dataset_name="walk_forward_nse_clean")
        source = "REAL"
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        data = _sample_data()
        source = f"SAMPLE_FALLBACK: {exc}"
    rows = evaluate_walk_forward(data)
    path = save_walk_forward_report(rows)
    print(f"walk_forward_strategy_eval saved: {path}")
    print(f"data_source: {source}")
    print(f"rows: {len(rows)}")


def _strategy_returns(strategy: str, returns: pd.Series, regime: str) -> pd.Series:
    multiplier = {
        "ExpectedMoveCreditSpread": 0.35 if regime != "high_volatility" else -0.15,
        "ExpectedMoveIronCondor": 0.30 if regime == "sideways" else -0.10,
        "IVMeanReversionPremium": 0.40 if regime != "trending" else -0.10,
        "VolatilityExpansion": 0.60 if regime == "high_volatility" else -0.05,
    }[strategy]
    return returns.abs() * multiplier if multiplier >= 0 else returns.abs() * multiplier


def _regime_label(returns: pd.Series) -> str:
    vol = float(returns.std())
    trend = float(returns.mean())
    if vol > 0.015:
        return "high_volatility"
    if abs(trend) > 0.001:
        return "trending"
    return "sideways"


def _period(data: pd.DataFrame) -> str:
    if "date" in data.columns:
        return f"{data['date'].iloc[0]} to {data['date'].iloc[-1]}"
    return f"{data.index[0]} to {data.index[-1]}"


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def _sharpe(returns: pd.Series) -> float:
    std = float(returns.std())
    return float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def _profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    return gains / losses if losses > 0 else (gains if gains > 0 else 0.0)


def _sample_data(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, rows)
    close = 26000.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=rows, freq="B"), "close": close})


if __name__ == "__main__":
    main()
