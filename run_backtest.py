"""Run an end-to-end BACKTEST workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import BacktestConfig, BacktestEngine, BacktestReportWriter
from market.data_loader import DataLoader
from news import NewsConfigLoader
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the backtest engine on historical or sample data."""

    parser = argparse.ArgumentParser(description="Run BACKTEST mode end-to-end")
    parser.add_argument("--sample", action="store_true", help="Force generated sample OHLCV data")
    parser.add_argument("--rows", type=int, default=180, help="Sample row count when --sample is used")
    parser.add_argument("--max-real-rows", type=int, default=500, help="Latest real rows to use unless --full-history is set")
    parser.add_argument("--full-history", action="store_true", help="Use all available historical rows")
    parser.add_argument("--report-dir", default="reports/backtests", help="Backtest report output directory")
    parser.add_argument("--dataset-name", default="final_backtest", help="Backtest dataset/report name")
    parser.add_argument("--use-ml-prediction", action="store_true", help="Enable ML prediction during simulation")
    parser.add_argument("--walk-forward", action="store_true", help="Run slower walk-forward validation")
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
    data_loader = DataLoader(config_loader=loader)
    historical_files = tuple(data_loader._discover_files())
    data_source = "SAMPLE"
    if args.sample:
        data = _sample_data(args.rows)
    elif historical_files:
        data_source = "REAL"
        try:
            data = _load_real_data(data_loader, args)
        except Exception as exc:
            print(f"Data source: REAL")
            print(f"Real data loading failed: {type(exc).__name__}: {exc}")
            return 1
    else:
        LOGGER.warning("No historical data files found; using generated sample backtest data")
        data = _sample_data(args.rows)

    print(f"Data source: {data_source}")
    result = engine.run(data, config=_backtest_config(args, data_source), save_report=True)

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


def _load_real_data(data_loader: DataLoader, args: argparse.Namespace) -> pd.DataFrame:
    """Load real NSE data for CLI backtests."""

    data, _report = data_loader.load_30_years_nse_data()
    if not args.full_history and args.max_real_rows > 0 and len(data) > args.max_real_rows:
        data = data.tail(args.max_real_rows).reset_index(drop=True)
        LOGGER.info("Using latest real historical rows", extra={"rows": len(data), "full_history": False})
    else:
        LOGGER.info("Using full real historical dataset", extra={"rows": len(data), "full_history": True})
    return data


def _backtest_config(args: argparse.Namespace, data_source: str) -> BacktestConfig:
    """Build CLI backtest configuration."""

    return BacktestConfig(
        dataset_name=f"{args.dataset_name}_{data_source}",
        use_ml_prediction=args.use_ml_prediction,
        run_walk_forward=args.walk_forward,
    )


if __name__ == "__main__":
    raise SystemExit(main())
