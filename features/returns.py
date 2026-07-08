"""Return and directional feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market.data_store import DataStore
from utils.exceptions import DataValidationError


def add_return_features(
    dataframe: pd.DataFrame,
    *,
    rolling_window: int = 20,
    probability_window: int = 20,
) -> pd.DataFrame:
    """Add daily, log, gap, intraday, rolling, and directional return features."""

    data = _prepare_ohlc(dataframe)
    data["previous_close"] = data["close"].shift(1)
    data["daily_return"] = data["close"].pct_change()
    data["log_return"] = np.log(data["close"] / data["previous_close"])
    data["overnight_gap"] = (data["open"] - data["previous_close"]) / data["previous_close"]
    data["intraday_return"] = (data["close"] - data["open"]) / data["open"]
    data["rolling_mean_return"] = data["daily_return"].rolling(rolling_window, min_periods=1).mean()
    data["rolling_return_std"] = data["daily_return"].rolling(rolling_window, min_periods=2).std()
    data["gap_up_probability"] = (
        (data["overnight_gap"] > 0).rolling(probability_window, min_periods=1).mean()
    )
    data["gap_down_probability"] = (
        (data["overnight_gap"] < 0).rolling(probability_window, min_periods=1).mean()
    )
    positive_returns = data["daily_return"].where(data["daily_return"] > 0)
    negative_returns = data["daily_return"].where(data["daily_return"] < 0)
    data["average_upside"] = positive_returns.rolling(probability_window, min_periods=1).mean().fillna(0.0)
    data["average_downside"] = negative_returns.rolling(probability_window, min_periods=1).mean().fillna(0.0)
    return data.drop(columns=["previous_close"])


def save_return_features(
    dataframe: pd.DataFrame,
    name: str = "return_features",
    *,
    data_store: DataStore | None = None,
) -> tuple[pd.DataFrame, object]:
    """Generate and save return features to the feature store."""

    features = add_return_features(dataframe)
    store = data_store or DataStore()
    path = store.save_features(features, name)
    return features, path


def _prepare_ohlc(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise DataValidationError(f"Missing OHLC columns: {sorted(missing)}")
    data = dataframe.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    if data.empty:
        raise DataValidationError("No valid OHLC rows available for feature generation")
    return data
