"""Configurable persistence for processed, feature, and model datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from utils.config_loader import ConfigLoader
from utils.exceptions import DataValidationError
from utils.file_utils import ensure_parent_dir

DatasetKind = Literal["processed", "features", "models"]


class DataStore:
    """Save and load tabular datasets using CSV and parquet where possible."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        processed_dir: str | Path | None = None,
        features_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
        """Create a data store using config paths unless explicit paths are supplied."""

        self.config_loader = config_loader or ConfigLoader()
        config = self.config_loader.load()
        self.processed_dir = Path(processed_dir or config.settings.data_paths["processed"])
        self.features_dir = Path(features_dir or config.settings.data_paths.get("features", Path("data/features")))
        self.models_dir = Path(models_dir or config.settings.data_paths["models"])

    def save_processed(self, dataframe: pd.DataFrame, name: str, *, prefer_parquet: bool = True) -> Path:
        """Save a processed dataset."""

        return self.save_dataset(dataframe, name, kind="processed", prefer_parquet=prefer_parquet)

    def load_processed(self, name: str) -> pd.DataFrame:
        """Load a processed dataset."""

        return self.load_dataset(name, kind="processed")

    def save_features(self, dataframe: pd.DataFrame, name: str, *, prefer_parquet: bool = True) -> Path:
        """Save a feature dataset."""

        return self.save_dataset(dataframe, name, kind="features", prefer_parquet=prefer_parquet)

    def load_features(self, name: str) -> pd.DataFrame:
        """Load a feature dataset."""

        return self.load_dataset(name, kind="features")

    def save_model_dataset(self, dataframe: pd.DataFrame, name: str, *, prefer_parquet: bool = True) -> Path:
        """Save a model-ready dataset."""

        return self.save_dataset(dataframe, name, kind="models", prefer_parquet=prefer_parquet)

    def load_model_dataset(self, name: str) -> pd.DataFrame:
        """Load a model-ready dataset."""

        return self.load_dataset(name, kind="models")

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
        name: str,
        *,
        kind: DatasetKind,
        prefer_parquet: bool = True,
    ) -> Path:
        """Save a dataframe by dataset kind."""

        if dataframe.empty:
            raise DataValidationError("Cannot save an empty dataset")
        base_path = self._base_dir(kind) / self._normalize_name(name)
        if prefer_parquet:
            parquet_path = base_path.with_suffix(".parquet")
            try:
                ensure_parent_dir(parquet_path)
                dataframe.to_parquet(parquet_path, index=False)
                return parquet_path
            except (ImportError, ValueError, ModuleNotFoundError):
                pass
        csv_path = base_path.with_suffix(".csv")
        ensure_parent_dir(csv_path)
        dataframe.to_csv(csv_path, index=False)
        return csv_path

    def load_dataset(self, name: str, *, kind: DatasetKind) -> pd.DataFrame:
        """Load a dataframe by dataset kind."""

        base_path = self._base_dir(kind) / self._normalize_name(name)
        parquet_path = base_path.with_suffix(".parquet")
        csv_path = base_path.with_suffix(".csv")
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except (ImportError, ValueError, ModuleNotFoundError):
                if not csv_path.exists():
                    raise
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise DataValidationError(f"Dataset not found: {base_path}")

    def _base_dir(self, kind: DatasetKind) -> Path:
        if kind == "processed":
            return self.processed_dir
        if kind == "features":
            return self.features_dir
        if kind == "models":
            return self.models_dir
        raise DataValidationError(f"Unsupported dataset kind: {kind}")

    @staticmethod
    def _normalize_name(name: str) -> str:
        cleaned = name.strip().replace("\\", "/").split("/")[-1]
        if not cleaned:
            raise DataValidationError("Dataset name is required")
        return Path(cleaned).stem
