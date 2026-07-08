# backtesting: event-driven backtesting engine
"""Backtesting engine, walk-forward validation, costs, metrics, and reports."""

from backtesting.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from backtesting.performance_metrics import PerformanceMetrics, PerformanceSummary, max_losing_streak
from backtesting.reports import BacktestReportWriter
from backtesting.transaction_costs import FillResult, FillSimulator, TransactionCostModel, approximate_option_payoff
from backtesting.walk_forward import WalkForwardSplitter, WalkForwardWindow, run_walk_forward_validation

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestReportWriter",
    "BacktestResult",
    "FillResult",
    "FillSimulator",
    "PerformanceMetrics",
    "PerformanceSummary",
    "TransactionCostModel",
    "WalkForwardSplitter",
    "WalkForwardWindow",
    "approximate_option_payoff",
    "max_losing_streak",
    "run_walk_forward_validation",
]
