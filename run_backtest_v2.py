"""Run the research StrategySelectorV2 backtest stack."""

from __future__ import annotations

import argparse
from pathlib import Path

from backtesting.backtest_engine_v2 import BacktestEngineV2, BacktestV2Config
from utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run V2 research strategy backtest.")
    parser.add_argument("--report-dir", default="reports/backtests")
    parser.add_argument("--debug-dir", default="reports/debug")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--all-rows", action="store_true", help="Use all available historical rows instead of the 500-row smoke default.")
    parser.add_argument("--no-stress", action="store_true")
    parser.add_argument("--sample", action="store_true", help="Explicitly use deterministic generated sample data.")
    parser.add_argument("--sample-rows", type=int, default=220, help="Generated row count used with --sample.")
    return parser.parse_args()


def main() -> None:
    """Run BacktestEngineV2."""

    args = parse_args()
    setup_logging()
    engine = BacktestEngineV2(report_dir=Path(args.report_dir), debug_dir=Path(args.debug_dir))
    max_rows = None if args.all_rows else args.max_rows
    result = engine.run(config=BacktestV2Config(max_rows=max_rows, stress_test_enabled=not args.no_stress, use_sample_data=args.sample, sample_rows=args.sample_rows))
    print(f"Data source: {result.summary['data_source']}")
    print(f"Backtest V2 report: {result.report_path}")
    print(f"Trades: {result.summary['trades']}")
    print(f"Net P&L: {result.summary['net_pnl']:.2f}")
    print(f"Debug reports: {result.summary.get('debug_reports', {})}")


if __name__ == "__main__":
    main()
