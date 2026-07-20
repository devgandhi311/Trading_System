"""Order and trade state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
import hashlib
import json
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

    @property
    def idempotency_key(self) -> str:
        """Return a stable key for one logical order intent.

        Callers may provide an explicit key in metadata for durable workflows.
        Otherwise the key is derived from immutable economic order fields; the
        wall-clock timestamp is deliberately excluded so a recreated intent is
        still recognized after a retry or restart.
        """

        explicit = self.metadata.get("idempotency_key")
        if explicit:
            return str(explicit)
        payload = {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "strategy_id": self.strategy_id,
            "limit_price": self.limit_price,
            "trigger_price": self.trigger_price,
            "product_type": self.product_type,
            "validity": self.validity,
            "tag": self.tag,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class OrderIntentStatus(StrEnum):
    """Local certainty about one idempotent broker submission."""

    IN_FLIGHT = "IN_FLIGHT"
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    RETRY_ALLOWED = "RETRY_ALLOWED"


@dataclass(frozen=True)
class OrderIntentRecord:
    """Tracked outcome for one logical submission intent."""

    idempotency_key: str
    status: OrderIntentStatus
    response: "OrderResponse | None" = None
    attempts: int = 0
    detail: str | None = None


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
