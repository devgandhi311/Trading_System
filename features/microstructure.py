"""Microstructure context features for regime, prediction, and filtering only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market.data_store import DataStore
from utils.exceptions import DataValidationError


def add_microstructure_features(
    dataframe: pd.DataFrame,
    *,
    rolling_window: int = 20,
    opening_range_bars: int = 5,
) -> pd.DataFrame:
    """Add non-signal microstructure features for regime and filtering use only."""

    data = _prepare_market_data(dataframe)
    spread = data["ask"] - data["bid"] if {"ask", "bid"}.issubset(data.columns) else np.nan
    midpoint = (data["ask"] + data["bid"]) / 2 if {"ask", "bid"}.issubset(data.columns) else data["close"]
    data["bid_ask_spread_percentage"] = spread / midpoint.replace(0, np.nan)
    data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3
    data["vwap"] = data.get("vwap", _rolling_vwap(data, rolling_window))
    data["vwap_deviation"] = (data["close"] - data["vwap"]) / data["vwap"].replace(0, np.nan)
    data["volume_mean"] = data["volume"].rolling(rolling_window, min_periods=1).mean()
    data["volume_std"] = data["volume"].rolling(rolling_window, min_periods=2).std().replace(0, np.nan)
    data["volume_spike"] = (data["volume"] - data["volume_mean"]) / data["volume_std"]
    data["opening_range_high"] = data["high"].rolling(opening_range_bars, min_periods=1).max()
    data["opening_range_low"] = data["low"].rolling(opening_range_bars, min_periods=1).min()
    data["opening_range_breakout"] = (
        (data["close"] > data["opening_range_high"].shift(1))
        | (data["close"] < data["opening_range_low"].shift(1))
    )
    data["previous_day_high"] = data["high"].shift(1)
    data["previous_day_low"] = data["low"].shift(1)
    data["previous_day_high_sweep"] = (data["high"] > data["previous_day_high"]) & (data["close"] < data["previous_day_high"])
    data["previous_day_low_sweep"] = (data["low"] < data["previous_day_low"]) & (data["close"] > data["previous_day_low"])
    data["bar_range"] = data["high"] - data["low"]
    data["average_range"] = data["bar_range"].rolling(rolling_window, min_periods=1).mean()
    data["range_expansion"] = data["bar_range"] > data["average_range"] * 1.5
    data["range_compression"] = data["bar_range"] < data["average_range"] * 0.5
    normalized_volume = _minmax(data["volume"].rolling(rolling_window, min_periods=1).mean())
    normalized_spread = 1.0 - _minmax(data["bid_ask_spread_percentage"].fillna(0.0))
    data["liquidity_score"] = ((normalized_volume + normalized_spread) / 2).clip(0.0, 1.0)
    data["slippage_estimate"] = (
        data["bid_ask_spread_percentage"].fillna(0.0) / 2
        + (1.0 - data["liquidity_score"]) * data["bar_range"].div(data["close"].replace(0, np.nan)).fillna(0.0)
    )
    body = (data["close"] - data["open"]).abs()
    upper_wick = data["high"] - data[["open", "close"]].max(axis=1)
    lower_wick = data[["open", "close"]].min(axis=1) - data["low"]
    range_zscore = (data["bar_range"] - data["average_range"]) / data["bar_range"].rolling(rolling_window, min_periods=2).std()
    data["abnormal_candle_detection"] = (
        (range_zscore > 2.0)
        | (upper_wick > body * 2.5)
        | (lower_wick > body * 2.5)
    )
    return data.drop(columns=["typical_price", "volume_mean", "volume_std"], errors="ignore")


def save_microstructure_features(
    dataframe: pd.DataFrame,
    name: str = "microstructure_features",
    *,
    data_store: DataStore | None = None,
) -> tuple[pd.DataFrame, object]:
    """Generate and save microstructure context features."""

    features = add_microstructure_features(dataframe)
    store = data_store or DataStore()
    path = store.save_features(features, name)
    return features, path


def _rolling_vwap(data: pd.DataFrame, rolling_window: int) -> pd.Series:
    value = data["typical_price"] * data["volume"]
    volume = data["volume"].rolling(rolling_window, min_periods=1).sum().replace(0, np.nan)
    return value.rolling(rolling_window, min_periods=1).sum() / volume


def _minmax(series: pd.Series) -> pd.Series:
    minimum = series.rolling(len(series), min_periods=1).min()
    maximum = series.rolling(len(series), min_periods=1).max()
    return ((series - minimum) / (maximum - minimum).replace(0, np.nan)).fillna(0.5)


def _prepare_market_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise DataValidationError(f"Missing market columns: {sorted(missing)}")
    data = dataframe.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    numeric_columns = ["open", "high", "low", "close", "volume", "bid", "ask", "vwap"]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close", "volume"])
    if data.empty:
        raise DataValidationError("No valid market rows available for microstructure features")
    return data
