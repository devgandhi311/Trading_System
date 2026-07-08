"""Shared enums used across the trading system."""

from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    """Supported runtime modes."""

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class OrderSide(StrEnum):
    """Order transaction side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL_LIMIT = "SL_LIMIT"


class OrderStatus(StrEnum):
    """Lifecycle states for broker and paper orders."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class StrategyType(StrEnum):
    """Known strategy families."""

    MOMENTUM_BREAKOUT = "momentum_breakout"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_SCALPING = "volatility_scalping"
    IRON_CONDOR = "iron_condor"
    CREDIT_SPREAD = "credit_spread"
    DEBIT_SPREAD = "debit_spread"
    LONG_GAMMA = "long_gamma"
    SHORT_STRANGLE = "short_strangle"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    CUSTOM = "custom"


class RegimeType(StrEnum):
    """Market regimes used by prediction and strategy selection."""

    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"
