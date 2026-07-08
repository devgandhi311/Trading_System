"""Strategy plan models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models.common import OrderSide, OrderType, StrategyType


@dataclass(frozen=True)
class StrategyLeg:
    """Single executable leg in a strategy plan."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    trigger_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyPlan:
    """Validated strategy intent before risk and execution checks."""

    strategy_id: str
    strategy_type: StrategyType
    symbol: str
    created_at: datetime
    legs: tuple[StrategyLeg, ...]
    confidence: float
    rationale: str
    max_loss: float | None = None
    expected_profit: float | None = None
    expiry_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
