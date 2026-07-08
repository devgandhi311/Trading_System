"""Approximate smart-money pattern context for regime, confidence, and filtering only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class PatternDetection:
    """Detected pattern context; not a direct trade trigger."""

    detected: bool
    strength: float
    explanation: str


def bullish_fair_value_gap(dataframe: pd.DataFrame) -> PatternDetection:
    """Detect a simple three-candle bullish fair value gap."""

    data = _prepare_ohlc(dataframe)
    if len(data) < 3:
        return _result(False, 0.0, "Need at least three candles for bullish FVG")
    previous_high = data.iloc[-3]["high"]
    current_low = data.iloc[-1]["low"]
    gap = current_low - previous_high
    atr = _average_range(data)
    return _result(gap > 0, gap / atr if atr else 0.0, "Current low is above the high two candles back")


def bearish_fair_value_gap(dataframe: pd.DataFrame) -> PatternDetection:
    """Detect a simple three-candle bearish fair value gap."""

    data = _prepare_ohlc(dataframe)
    if len(data) < 3:
        return _result(False, 0.0, "Need at least three candles for bearish FVG")
    previous_low = data.iloc[-3]["low"]
    current_high = data.iloc[-1]["high"]
    gap = previous_low - current_high
    atr = _average_range(data)
    return _result(gap > 0, gap / atr if atr else 0.0, "Current high is below the low two candles back")


def liquidity_sweep_above_previous_high(dataframe: pd.DataFrame) -> PatternDetection:
    """Detect sweep above prior high followed by close back below it."""

    data = _prepare_ohlc(dataframe)
    previous_high = data.iloc[:-1]["high"].max()
    last = data.iloc[-1]
    swept = last["high"] > previous_high and last["close"] < previous_high
    return _result(swept, (last["high"] - previous_high) / _average_range(data), "High swept prior liquidity and closed back below")


def liquidity_sweep_below_previous_low(dataframe: pd.DataFrame) -> PatternDetection:
    """Detect sweep below prior low followed by close back above it."""

    data = _prepare_ohlc(dataframe)
    previous_low = data.iloc[:-1]["low"].min()
    last = data.iloc[-1]
    swept = last["low"] < previous_low and last["close"] > previous_low
    return _result(swept, (previous_low - last["low"]) / _average_range(data), "Low swept prior liquidity and closed back above")


def bullish_order_block(dataframe: pd.DataFrame) -> PatternDetection:
    """Approximate bullish order block as last bearish candle before displacement up."""

    data = _prepare_ohlc(dataframe)
    if len(data) < 2:
        return _result(False, 0.0, "Need at least two candles for bullish order block")
    previous = data.iloc[-2]
    last = data.iloc[-1]
    displacement = _body(last) / _average_body(data)
    detected = previous["close"] < previous["open"] and last["close"] > last["open"] and displacement > 1.5
    return _result(detected, displacement / 3, "Bearish candle followed by bullish displacement")


def bearish_order_block(dataframe: pd.DataFrame) -> PatternDetection:
    """Approximate bearish order block as last bullish candle before displacement down."""

    data = _prepare_ohlc(dataframe)
    if len(data) < 2:
        return _result(False, 0.0, "Need at least two candles for bearish order block")
    previous = data.iloc[-2]
    last = data.iloc[-1]
    displacement = _body(last) / _average_body(data)
    detected = previous["close"] > previous["open"] and last["close"] < last["open"] and displacement > 1.5
    return _result(detected, displacement / 3, "Bullish candle followed by bearish displacement")


def break_of_structure(dataframe: pd.DataFrame, *, lookback: int = 5) -> PatternDetection:
    """Detect close beyond recent structure high or low."""

    data = _prepare_ohlc(dataframe)
    prior = data.iloc[-lookback - 1 : -1] if len(data) > lookback else data.iloc[:-1]
    if prior.empty:
        return _result(False, 0.0, "Need prior candles for structure break")
    last_close = data.iloc[-1]["close"]
    high_break = last_close > prior["high"].max()
    low_break = last_close < prior["low"].min()
    distance = max(last_close - prior["high"].max(), prior["low"].min() - last_close, 0)
    return _result(high_break or low_break, distance / _average_range(data), "Close broke recent structure")


def change_of_character(dataframe: pd.DataFrame, *, lookback: int = 5) -> PatternDetection:
    """Detect a structure break opposite the recent return direction."""

    data = _prepare_ohlc(dataframe)
    returns = data["close"].pct_change()
    recent_bias = returns.iloc[-lookback - 1 : -1].mean()
    bos = break_of_structure(data, lookback=lookback)
    if not bos.detected or pd.isna(recent_bias):
        return _result(False, 0.0, "No opposing structure change detected")
    last_close = data.iloc[-1]["close"]
    prior = data.iloc[-lookback - 1 : -1] if len(data) > lookback else data.iloc[:-1]
    opposing = (recent_bias > 0 and last_close < prior["low"].min()) or (recent_bias < 0 and last_close > prior["high"].max())
    return _result(opposing, bos.strength, "Structure break opposed recent directional bias")


def equal_highs(dataframe: pd.DataFrame, *, tolerance_percent: float = 0.001) -> PatternDetection:
    """Detect nearby equal highs."""

    data = _prepare_ohlc(dataframe)
    highs = data["high"].tail(5)
    distance = highs.max() - highs.min()
    tolerance = highs.mean() * tolerance_percent
    return _result(distance <= tolerance, 1 - distance / tolerance if tolerance else 0.0, "Recent highs are clustered")


def equal_lows(dataframe: pd.DataFrame, *, tolerance_percent: float = 0.001) -> PatternDetection:
    """Detect nearby equal lows."""

    data = _prepare_ohlc(dataframe)
    lows = data["low"].tail(5)
    distance = lows.max() - lows.min()
    tolerance = lows.mean() * tolerance_percent
    return _result(distance <= tolerance, 1 - distance / tolerance if tolerance else 0.0, "Recent lows are clustered")


def failed_breakout(dataframe: pd.DataFrame, *, lookback: int = 5) -> PatternDetection:
    """Detect breakout beyond recent range that closes back inside."""

    data = _prepare_ohlc(dataframe)
    prior = data.iloc[-lookback - 1 : -1] if len(data) > lookback else data.iloc[:-1]
    if prior.empty:
        return _result(False, 0.0, "Need prior candles for failed breakout")
    last = data.iloc[-1]
    prior_high = prior["high"].max()
    prior_low = prior["low"].min()
    failed_up = last["high"] > prior_high and last["close"] < prior_high
    failed_down = last["low"] < prior_low and last["close"] > prior_low
    distance = max(last["high"] - prior_high, prior_low - last["low"], 0)
    return _result(failed_up or failed_down, distance / _average_range(data), "Breakout failed and closed back inside range")


def displacement_candle(dataframe: pd.DataFrame, *, multiplier: float = 1.8) -> PatternDetection:
    """Detect candle body materially larger than recent average body."""

    data = _prepare_ohlc(dataframe)
    ratio = _body(data.iloc[-1]) / _average_body(data)
    return _result(ratio >= multiplier, ratio / (multiplier * 2), "Latest candle body displaced from recent average")


def _prepare_ohlc(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise DataValidationError(f"Missing OHLC columns: {sorted(missing)}")
    data = dataframe.copy().reset_index(drop=True)
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=list(required))
    if data.empty:
        raise DataValidationError("No valid OHLC rows for smart-money detection")
    return data


def _result(detected: bool, strength: float, explanation: str) -> PatternDetection:
    clean_strength = 0.0 if pd.isna(strength) else float(np.clip(strength, 0.0, 1.0))
    return PatternDetection(bool(detected), clean_strength if detected else 0.0, explanation)


def _average_range(data: pd.DataFrame) -> float:
    value = float((data["high"] - data["low"]).tail(20).mean())
    return value if value > 0 else 1.0


def _body(row: pd.Series) -> float:
    return float(abs(row["close"] - row["open"]))


def _average_body(data: pd.DataFrame) -> float:
    value = float((data["close"] - data["open"]).abs().tail(20).mean())
    return value if value > 0 else 1.0
