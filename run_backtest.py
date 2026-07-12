"""Run an end-to-end BACKTEST workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import pandas as pd

from backtesting import BacktestConfig, BacktestEngine, BacktestReportWriter
from market.data_loader import DataLoader, generate_sample_ohlcv
from models.market_models import DatasetManifest
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
    parser.add_argument("--minimum-rows", type=int, default=40, help="Minimum validated rows required to run")
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
    data_source = "SAMPLE" if args.sample else "REAL"
    if args.sample:
        data = _sample_data(args.rows)
        manifest = _sample_manifest(data, timezone_name=config.settings.timezone)
    else:
        try:
            data, manifest = _load_real_data(data_loader, args)
        except Exception as exc:
            print("Data source: REAL")
            print(f"Real data loading failed: {type(exc).__name__}: {exc}")
            return 1

    if len(data) < args.minimum_rows:
        print(f"Data source: {data_source}")
        print(f"Insufficient history: found {len(data)} rows; require at least {args.minimum_rows}")
        return 1
    print(f"Data source: {data_source}")
    result = engine.run(data, config=_backtest_config(args, data_source), save_report=True)
    manifest_path = _write_manifest(manifest, result.report_path)
    _embed_manifest(result.report_path, manifest)

    print(f"Backtest complete: trades={len(result.trades)}, total_return={result.metrics.total_return:.4f}")
    print(f"Report: {Path(result.report_path).resolve()}")
    print(f"Dataset manifest: {manifest_path.resolve()}")
    return 0


def _sample_data(rows: int) -> pd.DataFrame:
    """Generate deterministic NSE-like sample data for smoke tests."""

    return generate_sample_ohlcv(rows)


def _load_real_data(data_loader: DataLoader, args: argparse.Namespace) -> tuple[pd.DataFrame, DatasetManifest]:
    """Load real NSE data for CLI backtests."""

    data, quality_report = data_loader.load_30_years_nse_data()
    if not args.full_history and args.max_real_rows > 0 and len(data) > args.max_real_rows:
        data = data.tail(args.max_real_rows).reset_index(drop=True)
        LOGGER.info("Using latest real historical rows", extra={"rows": len(data), "full_history": False})
    else:
        LOGGER.info("Using full real historical dataset", extra={"rows": len(data), "full_history": True})
    manifest = quality_report.manifest
    if len(data) != manifest.row_count:
        manifest = replace(
            manifest,
            first_timestamp=pd.Timestamp(data["date"].min()).to_pydatetime(),
            last_timestamp=pd.Timestamp(data["date"].max()).to_pydatetime(),
            row_count=len(data),
        )
    return data, manifest


def _sample_manifest(data: pd.DataFrame, *, timezone_name: str = "Asia/Kolkata") -> DatasetManifest:
    """Build provenance for an explicitly requested generated sample."""

    return DatasetManifest(
        manifest_id=str(__import__("uuid").uuid4()),
        data_source="SAMPLE",
        source_type="SAMPLE",
        source_file_paths=(),
        source_file_hashes={},
        source_file_sizes={},
        detected_file_formats={},
        row_count=len(data),
        first_timestamp=pd.Timestamp(data["date"].min()).to_pydatetime(),
        last_timestamp=pd.Timestamp(data["date"].max()).to_pydatetime(),
        detected_columns=tuple(str(column) for column in data.columns),
        normalized_columns=tuple(str(column) for column in data.columns),
        removed_rows=0,
        duplicate_rows=0,
        invalid_rows=0,
        missing_value_counts={str(column): int(data[column].isna().sum()) for column in data.columns},
        timezone=timezone_name,
        generated_sample=True,
        loader_version=DataLoader.LOADER_VERSION,
        creation_timestamp=datetime.now(timezone.utc),
    )


def _write_manifest(manifest: DatasetManifest, report_path: Path) -> Path:
    """Write a dataset manifest beside the primary backtest report."""

    path = report_path.with_name(f"{report_path.stem}_dataset_manifest.json")
    path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _embed_manifest(report_path: Path, manifest: DatasetManifest) -> None:
    """Append manifest metadata to a supported Excel backtest report."""

    if report_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        return
    rows = [{"field": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value} for key, value in manifest.as_dict().items()]
    with pd.ExcelWriter(report_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Dataset_Manifest", index=False)


def _backtest_config(args: argparse.Namespace, data_source: str) -> BacktestConfig:
    """Build CLI backtest configuration."""

    return BacktestConfig(
        dataset_name=f"{args.dataset_name}_{data_source}",
        use_ml_prediction=args.use_ml_prediction,
        run_walk_forward=args.walk_forward,
    )


if __name__ == "__main__":
    raise SystemExit(main())
