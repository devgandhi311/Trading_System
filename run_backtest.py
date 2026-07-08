"""Run an end-to-end BACKTEST workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import BacktestConfig, BacktestEngine, BacktestReportWriter
from news import NewsConfigLoader
from utils.config_loader import ConfigLoader
from utils.exceptions import DataValidationError
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the backtest engine on historical or sample data."""

    parser = argparse.ArgumentParser(description="Run BACKTEST mode end-to-end")
    parser.add_argument("--sample", action="store_true", help="Use generated sample OHLCV data")
    parser.add_argument("--rows", type=int, default=180, help="Sample row count when --sample is used")
    parser.add_argument("--report-dir", default="reports/backtests", help="Backtest report output directory")
    parser.add_argument("--dataset-name", default="final_backtest", help="Backtest dataset/report name")
    args = parser.parse_args(argv)

    loader = ConfigLoader()
    config = loader.load()
    setup_logging(level=config.settings.log_level)
    news_config = NewsConfigLoader().load()
    LOGGER.info(
        "Backtest news integration initialized",
        extra={"enabled": news_config.enabled, "mode": "cached_or_disabled_no_future_news"},
    )
    engine = BacktestEngine(
        config_loader=loader,
        report_writer=BacktestReportWriter(report_dir=args.report_dir),
    )
    data = _sample_data(args.rows) if args.sample else None
    if data is None:
        try:
            result = engine.run(config=BacktestConfig(dataset_name=args.dataset_name), save_report=True)
        except DataValidationError:
            LOGGER.warning("Historical data unavailable; falling back to generated sample backtest data")
            data = _sample_data(args.rows)
            result = engine.run(data, config=BacktestConfig(dataset_name=args.dataset_name), save_report=True)
    else:
        result = engine.run(data, config=BacktestConfig(dataset_name=args.dataset_name), save_report=True)

    print(f"Backtest complete: trades={len(result.trades)}, total_return={result.metrics.total_return:.4f}")
    print(f"Report: {Path(result.report_path).resolve()}")
    return 0


def _sample_data(rows: int) -> pd.DataFrame:
    """Generate deterministic NSE-like sample data for smoke tests."""

    count = max(rows, 40)
    dates = pd.date_range("2024-01-01", periods=count, freq="B")
    close = 20_000 + np.linspace(0, 500, count) + np.sin(np.arange(count) / 4) * 35
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 8,
            "high": close + 35,
            "low": close - 35,
            "close": close,
            "volume": np.full(count, 1_000_000),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
