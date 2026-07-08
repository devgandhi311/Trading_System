"""Market data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class MarketBar:
    """OHLCV market bar for a symbol and timeframe."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "1m"
    open_interest: int | None = None
    vwap: float | None = None


@dataclass(frozen=True)
class OptionContract:
    """Option contract metadata and latest quote fields."""

    symbol: str
    underlying: str
    expiry: date
    strike: float
    option_type: str
    lot_size: int
    exchange: str = "NSE"
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionChainSnapshot:
    """Point-in-time option chain snapshot for an underlying."""

    underlying: str
    timestamp: datetime
    spot_price: float
    expiry: date
    contracts: tuple[OptionContract, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
