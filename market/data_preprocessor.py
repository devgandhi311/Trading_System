"""Preprocessing helpers for continuous clean index-level datasets."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils.exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)


class DataPreprocessor:
    """Normalize dates, volumes, gaps, and anomalous index OHLC data."""

    def __init__(self, *, anomaly_return_threshold: float = 0.25) -> None:
        """Create a preprocessor with a daily return anomaly threshold."""

        if anomaly_return_threshold <= 0:
            raise ValueError("anomaly_return_threshold must be positive")
        self.anomaly_return_threshold = anomaly_return_threshold

    def preprocess(self, dataframe: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
        """Create a continuous clean index-level dataset."""

        if dataframe.empty:
            raise DataValidationError("Cannot preprocess an empty dataframe")
        data = dataframe.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None)
        data = data.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        data = self.handle_corporate_action_like_anomalies(data)
        data = self.handle_missing_trading_days(data)
        data = self.normalize_volume(data)
        if symbol is not None:
            data["symbol"] = symbol
        return data.reset_index(drop=True)

    def handle_missing_trading_days(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Reindex to business days and forward-fill index-level gaps."""

        if dataframe.empty:
            return dataframe
        data = dataframe.set_index("date").sort_index()
        business_index = pd.bdate_range(data.index.min(), data.index.max())
        data = data.reindex(business_index)
        price_columns = ["open", "high", "low", "close"]
        data[price_columns] = data[price_columns].ffill()
        if "volume" in data:
            data["volume"] = data["volume"].fillna(0)
        data.index.name = "date"
        return data.reset_index()

    def handle_corporate_action_like_anomalies(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Smooth extreme one-day index jumps that look like data adjustment anomalies."""

        data = dataframe.copy()
        returns = data["close"].pct_change().abs()
        anomalous = returns > self.anomaly_return_threshold
        if anomalous.any():
            LOGGER.warning("Adjusting corporate-action-like anomalies", extra={"count": int(anomalous.sum())})
            for index in data.index[anomalous]:
                previous_index = data.index.get_loc(index) - 1
                if previous_index < 0:
                    continue
                previous_close = data.iloc[previous_index]["close"]
                ratio = previous_close / data.at[index, "close"] if data.at[index, "close"] else 1.0
                for column in ("open", "high", "low", "close"):
                    data.at[index, column] = data.at[index, column] * ratio
        return data

    @staticmethod
    def normalize_volume(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add normalized volume columns while preserving raw volume."""

        data = dataframe.copy()
        if "volume" not in data:
            data["volume"] = 0
        data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0)
        data["volume_raw"] = data["volume"]
        mean = data["volume"].mean()
        std = data["volume"].std(ddof=0)
        if std == 0 or np.isnan(std):
            data["volume_zscore"] = 0.0
        else:
            data["volume_zscore"] = (data["volume"] - mean) / std
        max_volume = data["volume"].max()
        data["volume_normalized"] = 0.0 if max_volume == 0 else data["volume"] / max_volume
        return data
