"""Broker account and market data snapshot models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models.common import OrderSide, OrderStatus


@dataclass(frozen=True)
class BrokerProfile:
    """Non-secret broker profile information."""

    user_id: str
    broker_name: str
    account_type: str
    display_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarginSnapshot:
    """Available and utilized margin snapshot."""

    timestamp: datetime
    cash: float
    available_margin: float
    used_margin: float
    opening_balance: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionSnapshot:
    """Current broker position snapshot."""

    symbol: str
    quantity: int
    average_price: float
    last_price: float
    unrealized_pnl: float
    product_type: str = "MIS"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderSnapshot:
    """Read-only broker order snapshot."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    status: OrderStatus
    timestamp: datetime
    average_price: float | None = None
    filled_quantity: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HoldingSnapshot:
    """Read-only broker holding snapshot."""

    symbol: str
    quantity: int
    average_price: float
    last_price: float
    unrealized_pnl: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuoteSnapshot:
    """Market quote snapshot from broker or safe mock data."""

    symbol_or_token: str
    timestamp: datetime
    last_price: float
    bid: float | None = None
    ask: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
