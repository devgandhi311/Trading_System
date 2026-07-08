"""Regime-specific feature table creation for ML and regime detectors."""

from __future__ import annotations

import pandas as pd

from utils.exceptions import DataValidationError

REGIME_FEATURE_COLUMNS = (
    "date",
    "daily_return",
    "log_return",
    "overnight_gap",
    "intraday_return",
    "rolling_mean_return",
    "rolling_return_std",
    "rolling_volatility",
    "realized_volatility",
    "atr",
    "volatility_percentile",
    "range_percentage",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rolling_skewness",
    "rolling_kurtosis",
    "expected_move",
    "gap_up_probability",
    "gap_down_probability",
    "average_upside",
    "average_downside",
    "bid_ask_spread_percentage",
    "volume_spike",
    "vwap_deviation",
    "opening_range_breakout",
    "previous_day_high_sweep",
    "previous_day_low_sweep",
    "range_expansion",
    "range_compression",
    "liquidity_score",
    "slippage_estimate",
    "abnormal_candle_detection",
    "smart_money_context_strength",
)


def create_regime_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    """Create a regime-detector-safe table from combined features."""

    if features.empty:
        raise DataValidationError("Cannot create regime feature table from empty features")
    strength_columns = [column for column in features.columns if column.endswith("_strength")]
    detected_columns = [column for column in features.columns if column.endswith("_detected")]
    selected = list(
        dict.fromkeys(
            column
            for column in (*REGIME_FEATURE_COLUMNS, *strength_columns, *detected_columns)
            if column in features.columns
        )
    )
    if "date" not in selected:
        raise DataValidationError("Regime features require a date column")
    table = features[selected].copy()
    bool_columns = table.select_dtypes(include=["bool"]).columns
    table[bool_columns] = table[bool_columns].astype(int)
    numeric_columns = [column for column in table.columns if column != "date"]
    table[numeric_columns] = table[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return table
