"""Load, validate, clean, and persist long-horizon NSE OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from market.data_preprocessor import DataPreprocessor
from market.data_store import DataStore
from utils.config_loader import ConfigLoader
from utils.exceptions import DataValidationError
from utils.file_utils import ensure_parent_dir

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataQualityReport:
    """Quality statistics for a loaded NSE dataset."""

    source_files: int
    raw_rows: int
    clean_rows: int
    removed_bad_ohlc_rows: int
    removed_missing_rows: int
    removed_duplicate_rows: int
    output_path: Path
    report_path: Path


class DataLoader:
    """Load 30 years NSE OHLCV data and persist cleaned datasets."""

    COLUMN_ALIASES = {
        "date": {"date", "datetime", "timestamp", "time", "tradedate", "trade_date"},
        "open": {"open", "openprice", "open_price", "o"},
        "high": {"high", "highprice", "high_price", "h"},
        "low": {"low", "lowprice", "low_price", "l"},
        "close": {"close", "closeprice", "close_price", "last", "ltp", "c"},
        "volume": {"volume", "vol", "tradedqty", "traded_quantity", "qty", "sharestraded", "shares_traded"},
    }
    SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet"}

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        data_store: DataStore | None = None,
        preprocessor: DataPreprocessor | None = None,
        raw_dir: str | Path | None = None,
        report_path: str | Path | None = None,
    ) -> None:
        """Create a loader using config paths unless explicit paths are supplied."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.data_store = data_store or DataStore(config_loader=self.config_loader)
        self.preprocessor = preprocessor or DataPreprocessor()
        self.raw_dir = Path(raw_dir or self.config.settings.data_paths["raw"] / "nse_30_years")
        self.report_path = Path(
            report_path or Path("reports/risk_reports/data_quality_report.xlsx")
        )

    def load_30_years_nse_data(
        self,
        *,
        dataset_name: str = "nse_30_years_clean",
        preprocess: bool = True,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """Load raw NSE files, clean data, save output, and write quality report."""

        files = self._discover_files()
        if not files:
            raise DataValidationError(f"No NSE data files found in {self.raw_dir}")
        frames = [self._read_file(path) for path in files]
        raw = pd.concat(frames, ignore_index=True)
        clean, stats = self.clean(raw)
        if preprocess:
            clean = self.preprocessor.preprocess(clean)
        output_path = self.data_store.save_processed(clean, dataset_name)
        report = DataQualityReport(
            source_files=len(files),
            raw_rows=len(raw),
            clean_rows=len(clean),
            removed_bad_ohlc_rows=stats["removed_bad_ohlc_rows"],
            removed_missing_rows=stats["removed_missing_rows"],
            removed_duplicate_rows=stats["removed_duplicate_rows"],
            output_path=output_path,
            report_path=self.report_path,
        )
        self.save_quality_report(report)
        LOGGER.info("NSE data loaded", extra={"rows": len(clean), "files": len(files)})
        return clean, report

    def clean(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Normalize columns, validate OHLC, remove bad rows/duplicates, and sort."""

        data = self._standardize_columns(dataframe)
        before_missing = len(data)
        data = data.dropna(subset=["date", "open", "high", "low", "close"])
        removed_missing = before_missing - len(data)
        for column in ("open", "high", "low", "close", "volume"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        before_numeric = len(data)
        data = data.dropna(subset=["open", "high", "low", "close"])
        removed_missing += before_numeric - len(data)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        before_date = len(data)
        data = data.dropna(subset=["date"])
        removed_missing += before_date - len(data)

        valid_ohlc = (
            (data["high"] >= data["open"])
            & (data["high"] >= data["close"])
            & (data["high"] >= data["low"])
            & (data["low"] <= data["open"])
            & (data["low"] <= data["close"])
            & (data["low"] <= data["high"])
        )
        removed_bad_ohlc = int((~valid_ohlc).sum())
        data = data.loc[valid_ohlc].copy()
        before_duplicates = len(data)
        data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        removed_duplicates = before_duplicates - len(data)
        data["volume"] = data["volume"].fillna(0).clip(lower=0)
        return (
            data[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True),
            {
                "removed_bad_ohlc_rows": removed_bad_ohlc,
                "removed_missing_rows": removed_missing,
                "removed_duplicate_rows": removed_duplicates,
            },
        )

    def save_quality_report(self, report: DataQualityReport) -> Path:
        """Save data quality report to Excel."""

        ensure_parent_dir(report.report_path)
        rows = [
            {"metric": "source_files", "value": report.source_files},
            {"metric": "raw_rows", "value": report.raw_rows},
            {"metric": "clean_rows", "value": report.clean_rows},
            {"metric": "removed_bad_ohlc_rows", "value": report.removed_bad_ohlc_rows},
            {"metric": "removed_missing_rows", "value": report.removed_missing_rows},
            {"metric": "removed_duplicate_rows", "value": report.removed_duplicate_rows},
            {"metric": "output_path", "value": str(report.output_path)},
        ]
        pd.DataFrame(rows).to_excel(report.report_path, index=False)
        return report.report_path

    def _discover_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            return []
        return sorted(
            path
            for path in self.raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_SUFFIXES
        )

    def _read_file(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise DataValidationError(f"Unsupported file format: {path}")

    def _standardize_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized_columns = {
            self._normalize_column_name(column): column for column in dataframe.columns
        }
        mapping: dict[str, str] = {}
        for canonical, aliases in self.COLUMN_ALIASES.items():
            source = next((normalized_columns[alias] for alias in aliases if alias in normalized_columns), None)
            if source is None:
                raise DataValidationError(f"Missing required column: {canonical}")
            mapping[source] = canonical
        return dataframe.rename(columns=mapping)[["date", "open", "high", "low", "close", "volume"]].copy()

    @staticmethod
    def _normalize_column_name(value: Any) -> str:
        return "".join(character for character in str(value).lower() if character.isalnum() or character == "_")
