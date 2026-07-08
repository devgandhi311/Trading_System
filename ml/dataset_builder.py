"""Build supervised ML datasets from cached feature data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from market.data_store import DataStore
from utils.config_loader import ConfigLoader
from utils.exceptions import DataValidationError
from utils.file_utils import ensure_parent_dir


@dataclass(frozen=True)
class WalkForwardSplit:
    """Train/test indices for one walk-forward fold."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class DatasetBuildResult:
    """Built dataset paths and split metadata."""

    dataset: pd.DataFrame
    dataset_path: Path
    summary_report_path: Path
    splits: tuple[WalkForwardSplit, ...]


class DatasetBuilder:
    """Create supervised datasets from feature tables without look-ahead bias."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        data_store: DataStore | None = None,
        feature_dir: str | Path | None = None,
        model_data_dir: str | Path | None = None,
        report_dir: str | Path = "reports/model_reports",
    ) -> None:
        """Create a dataset builder."""

        self.config_loader = config_loader or ConfigLoader()
        config = self.config_loader.load()
        self.data_store = data_store or DataStore(config_loader=self.config_loader)
        self.feature_dir = Path(feature_dir or config.settings.data_paths.get("features", "data/features"))
        self.model_data_dir = Path(model_data_dir or config.settings.data_paths["models"])
        self.report_dir = Path(report_dir)

    def build_dataset(
        self,
        *,
        dataset_name: str = "supervised_dataset",
        train_window: int = 252,
        test_window: int = 63,
    ) -> DatasetBuildResult:
        """Load feature data, create targets, save dataset, and write report."""

        features = self.load_feature_data()
        dataset = self.create_targets(features)
        path = self.data_store.save_model_dataset(dataset, dataset_name)
        splits = self.walk_forward_splits(dataset, train_window=train_window, test_window=test_window)
        report_path = self.generate_summary_report(dataset, splits, dataset_name=dataset_name)
        return DatasetBuildResult(dataset=dataset, dataset_path=path, summary_report_path=report_path, splits=splits)

    def load_feature_data(self) -> pd.DataFrame:
        """Load all CSV/parquet feature datasets from data/features."""

        files = sorted(
            path for path in self.feature_dir.rglob("*") if path.suffix.lower() in {".csv", ".parquet"}
        )
        if not files:
            raise DataValidationError(f"No feature datasets found in {self.feature_dir}")
        frames = []
        for path in files:
            frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
        if "date" in data:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data.dropna(subset=["date"]).sort_values("date")
        return data.reset_index(drop=True)

    def create_targets(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create supervised targets using next-row values only."""

        required = {"close", "high", "low"}
        missing = required.difference(features.columns)
        if missing:
            raise DataValidationError(f"Missing columns for targets: {sorted(missing)}")
        data = features.copy().sort_values("date") if "date" in features else features.copy()
        next_close = data["close"].shift(-1)
        next_high = data["high"].shift(-1)
        next_low = data["low"].shift(-1)
        next_range = (next_high - next_low) / data["close"].replace(0, pd.NA)
        current_range = (data["high"] - data["low"]) / data["close"].replace(0, pd.NA)
        next_return = next_close / data["close"].replace(0, pd.NA) - 1
        data["target_next_day_return"] = next_return
        data["target_next_day_direction"] = (next_return > 0).astype(int)
        data["target_expected_range"] = next_range
        data["target_volatility_expansion"] = (next_range > current_range).astype(int)
        failure_source = data.get("failed_breakout_detected", pd.Series(False, index=data.index))
        data["target_breakout_failure"] = failure_source.shift(-1).fillna(False).astype(int)
        if "regime_name" in data:
            data["target_regime_label"] = data["regime_name"].shift(-1)
        elif "regime_label" in data:
            data["target_regime_label"] = data["regime_label"].shift(-1)
        else:
            data["target_regime_label"] = "unknown"
        return data.dropna(subset=["target_next_day_return", "target_expected_range"]).reset_index(drop=True)

    @staticmethod
    def walk_forward_splits(
        dataset: pd.DataFrame,
        *,
        train_window: int = 252,
        test_window: int = 63,
    ) -> tuple[WalkForwardSplit, ...]:
        """Create walk-forward train/test splits."""

        if train_window <= 0 or test_window <= 0:
            raise ValueError("train_window and test_window must be positive")
        splits = []
        fold = 0
        start = 0
        while start + train_window + test_window <= len(dataset):
            splits.append(
                WalkForwardSplit(
                    fold=fold,
                    train_start=start,
                    train_end=start + train_window,
                    test_start=start + train_window,
                    test_end=start + train_window + test_window,
                )
            )
            fold += 1
            start += test_window
        return tuple(splits)

    def generate_summary_report(
        self,
        dataset: pd.DataFrame,
        splits: tuple[WalkForwardSplit, ...],
        *,
        dataset_name: str,
    ) -> Path:
        """Generate an Excel dataset summary report."""

        path = ensure_parent_dir(self.report_dir / f"{dataset_name}_summary.xlsx")
        target_columns = [column for column in dataset.columns if column.startswith("target_")]
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(
                [
                    {"metric": "rows", "value": len(dataset)},
                    {"metric": "columns", "value": len(dataset.columns)},
                    {"metric": "target_columns", "value": ", ".join(target_columns)},
                    {"metric": "walk_forward_folds", "value": len(splits)},
                ]
            ).to_excel(writer, sheet_name="summary", index=False)
            pd.DataFrame([split.__dict__ for split in splits]).to_excel(writer, sheet_name="splits", index=False)
            dataset[target_columns].describe(include="all").reset_index().to_excel(
                writer, sheet_name="targets", index=False
            )
        return path
