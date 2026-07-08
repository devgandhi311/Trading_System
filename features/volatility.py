"""Volatility and range feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market.data_store import DataStore
from utils.exceptions import DataValidationError

TRADING_DAYS = 252


def add_volatility_features(
    dataframe: pd.DataFrame,
    *,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Add rolling, realized, ATR, distribution, and estimator volatility features."""

    data = _prepare_ohlcv(dataframe)
    previous_close = data["close"].shift(1)
    daily_return = data["close"].pct_change()
    log_hl = np.log(data["high"] / data["low"])
    log_co = np.log(data["close"] / data["open"])
    log_ho = np.log(data["high"] / data["open"])
    log_lo = np.log(data["low"] / data["open"])
    log_hc = np.log(data["high"] / data["close"])
    log_lc = np.log(data["low"] / data["close"])

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["rolling_volatility"] = daily_return.rolling(rolling_window, min_periods=2).std()
    data["realized_volatility"] = data["rolling_volatility"] * np.sqrt(TRADING_DAYS)
    data["atr"] = true_range.rolling(rolling_window, min_periods=1).mean()
    data["range_percentage"] = (data["high"] - data["low"]) / data["close"]
    parkinson_variance = (log_hl**2) / (4 * np.log(2))
    data["parkinson_volatility"] = np.sqrt(
        parkinson_variance.rolling(rolling_window, min_periods=1).mean() * TRADING_DAYS
    )
    garman_klass_variance = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
    data["garman_klass_volatility"] = np.sqrt(
        garman_klass_variance.clip(lower=0).rolling(rolling_window, min_periods=1).mean() * TRADING_DAYS
    )
    data["rolling_skewness"] = daily_return.rolling(rolling_window, min_periods=min(3, rolling_window)).skew()
    kurtosis_window = max(4, rolling_window)
    data["rolling_kurtosis"] = daily_return.rolling(kurtosis_window, min_periods=4).kurt()
    data["expected_move"] = data["close"] * data["realized_volatility"] / np.sqrt(TRADING_DAYS)
    data["volatility_percentile"] = data["realized_volatility"].rolling(
        rolling_window,
        min_periods=1,
    ).apply(_last_value_percentile, raw=False)
    data["garman_klass_intraday_component"] = (log_ho * log_hc + log_lo * log_lc).replace([np.inf, -np.inf], np.nan)
    return data


def save_volatility_features(
    dataframe: pd.DataFrame,
    name: str = "volatility_features",
    *,
    data_store: DataStore | None = None,
) -> tuple[pd.DataFrame, object]:
    """Generate and save volatility features to the feature store."""

    features = add_volatility_features(dataframe)
    store = data_store or DataStore()
    path = store.save_features(features, name)
    return features, path


def _last_value_percentile(values: pd.Series) -> float:
    if values.empty or pd.isna(values.iloc[-1]):
        return np.nan
    return float((values <= values.iloc[-1]).mean())


def _prepare_ohlcv(dataframe: pd.DataFrame) -> pd.DataFrame:
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
        raise DataValidationError("No valid OHLC rows available for volatility features")
    return data
