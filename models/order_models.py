"""Order and trade state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models.common import OrderSide, OrderStatus, OrderType


@dataclass(frozen=True)
class OrderRequest:
    """Order request sent to paper or broker execution."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    timestamp: datetime
    strategy_id: str | None = None
    limit_price: float | None = None
    trigger_price: float | None = None
    product_type: str = "MIS"
    validity: str = "DAY"
    tag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_emergency_exit(self) -> bool:
        """Return true when this request is explicitly marked as an emergency exit."""

        return bool(self.metadata.get("emergency_exit"))


@dataclass(frozen=True)
class OrderResponse:
    """Broker or paper broker response for an order request."""

    order_id: str
    status: OrderStatus
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int = 0
    average_price: float | None = None
    rejection_reason: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeState:
    """Current lifecycle state for an open or closed trade."""

    trade_id: str
    symbol: str
    side: OrderSide
    quantity: int
    entry_time: datetime
    entry_price: float
    status: OrderStatus
    strategy_id: str | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
