"""Tests for backtesting engine and analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestReportWriter,
    FillSimulator,
    PerformanceMetrics,
    TransactionCostModel,
    WalkForwardSplitter,
    approximate_option_payoff,
)
from models import OrderSide


def _historical_data(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    base = 20_000 + np.linspace(0, 400, rows) + np.sin(np.arange(rows) / 3) * 20
    close = base
    open_ = close - 5
    high = close + 30
    low = close - 30
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(rows, 1_000_000),
        }
    )


def test_transaction_costs_fill_and_option_payoff() -> None:
    costs = TransactionCostModel(slippage_percent=0.1)
    simulator = FillSimulator()

    assert costs.estimate(price=100.0, quantity=75, orders=2) > 40.0
    assert costs.apply_slippage(price=100.0, side=OrderSide.BUY) > 100.0
    assert simulator.limit_fill(side=OrderSide.BUY, limit_price=100.0, high=102.0, low=99.0).filled
    payoff = approximate_option_payoff(
        spot_price=20_200.0,
        strike=20_000.0,
        premium=100.0,
        option_type="CE",
        side=OrderSide.BUY,
        quantity=75,
    )
    assert payoff == 7_500.0


def test_performance_metrics_include_required_fields() -> None:
    trades = pd.DataFrame(
        {
            "net_pnl": [1000.0, -500.0, 750.0],
            "strategy_type": ["a", "a", "b"],
            "regime": ["bull", "bear", "bull"],
        }
    )
    equity = pd.DataFrame({"equity": [100_000.0, 101_000.0, 100_500.0, 101_250.0]})

    summary = PerformanceMetrics().calculate(trades, equity, initial_capital=100_000.0)

    assert summary.total_return > 0
    assert summary.win_rate == 2 / 3
    assert summary.profit_factor > 1
    assert summary.max_losing_streak == 1
    assert summary.strategy_wise_pnl["a"] == 500.0
    assert summary.regime_wise_pnl["bull"] == 1750.0


def test_walk_forward_splitter_creates_windows() -> None:
    data = _historical_data(40)
    windows = WalkForwardSplitter(train_size=20, test_size=5).split(data)

    assert len(windows) >= 3
    assert len(windows[0].train(data)) == 20
    assert len(windows[0].test(data)) == 5


def test_backtest_engine_runs_and_writes_report(tmp_path) -> None:
    engine = BacktestEngine(report_writer=BacktestReportWriter(report_dir=tmp_path))
    result = engine.run(
        _historical_data(80),
        config=BacktestConfig(
            initial_capital=100_000.0,
            quantity=25,
            rolling_window=10,
            max_holding_bars=3,
            use_ml_prediction=False,
            dataset_name="unit_backtest",
        ),
        save_report=True,
    )

    assert not result.equity_curve.empty
    assert result.report_path.exists()
    assert "net_pnl" in result.trades.columns
    assert isinstance(result.metrics.strategy_wise_pnl, dict)
    assert not result.walk_forward.empty


def test_backtest_engine_supports_no_report_mode() -> None:
    result = BacktestEngine().run(
        _historical_data(50),
        config=BacktestConfig(
            initial_capital=100_000.0,
            quantity=25,
            rolling_window=8,
            max_holding_bars=2,
            use_ml_prediction=False,
            dataset_name="unit_backtest_no_report",
        ),
        save_report=False,
    )

    assert str(result.report_path) == "."
    assert "equity" in result.equity_curve.columns
