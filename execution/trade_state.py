"""Execution trade-state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from models import OrderResponse, OrderSide, OrderStatus, StrategyPlan
from utils.time_utils import now_ist


@dataclass
class LegExecutionState:
    """Execution state for one strategy leg."""

    symbol: str
    side: OrderSide
    quantity: int
    order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float | None = None
    last_price: float | None = None

    @property
    def is_filled(self) -> bool:
        """Return true when the leg is completely filled."""

        return self.status == OrderStatus.FILLED and self.filled_quantity >= self.quantity

    @property
    def is_open(self) -> bool:
        """Return true when the leg has live quantity."""

        return self.status in {OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}


@dataclass
class ExecutionTradeState:
    """Track order, fill, and P&L state for a StrategyPlan execution."""

    strategy_id: str
    symbol: str
    created_at: datetime
    legs: dict[str, LegExecutionState] = field(default_factory=dict)
    stop_loss_order_ids: list[str] = field(default_factory=list)
    profit_order_ids: list[str] = field(default_factory=list)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    exit_reason: str | None = None
    manual_squareoff_requested: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_plan(cls, plan: StrategyPlan) -> "ExecutionTradeState":
        """Create state from a StrategyPlan."""

        state = cls(strategy_id=plan.strategy_id, symbol=plan.symbol, created_at=now_ist())
        for index, leg in enumerate(plan.legs):
            key = f"{index}:{leg.symbol}:{leg.side.value}"
            state.legs[key] = LegExecutionState(symbol=leg.symbol, side=leg.side, quantity=leg.quantity)
        return state

    def apply_order_response(self, response: OrderResponse) -> None:
        """Apply a broker/paper order response to matching leg state."""

        target = self._find_leg(response.symbol, response.side)
        if target is None:
            target = LegExecutionState(response.symbol, response.side, response.quantity)
            self.legs[f"response:{response.order_id}"] = target
        target.order_id = response.order_id
        target.status = response.status
        target.filled_quantity = response.filled_quantity
        target.average_price = response.average_price

    def update_last_price(self, symbol: str, last_price: float) -> None:
        """Update last price and recompute unrealized P&L."""

        for leg in self.legs.values():
            if leg.symbol == symbol:
                leg.last_price = last_price
        self.unrealized_pnl = self.calculate_unrealized_pnl()

    def calculate_unrealized_pnl(self) -> float:
        """Calculate simple unrealized P&L from filled legs."""

        pnl = 0.0
        for leg in self.legs.values():
            if leg.filled_quantity <= 0 or leg.average_price is None or leg.last_price is None:
                continue
            if leg.side == OrderSide.BUY:
                pnl += (leg.last_price - leg.average_price) * leg.filled_quantity
            else:
                pnl += (leg.average_price - leg.last_price) * leg.filled_quantity
        return pnl

    def filled_legs(self) -> tuple[LegExecutionState, ...]:
        """Return legs with any fill."""

        return tuple(leg for leg in self.legs.values() if leg.filled_quantity > 0)

    def has_partial_fill(self) -> bool:
        """Return true when some but not all legs are filled."""

        filled = [leg for leg in self.legs.values() if leg.filled_quantity > 0]
        return bool(filled) and len(filled) < len(self.legs)

    def all_entries_filled(self) -> bool:
        """Return true when all plan legs are fully filled."""

        return bool(self.legs) and all(leg.is_filled for leg in self.legs.values())

    def _find_leg(self, symbol: str, side: OrderSide) -> LegExecutionState | None:
        for leg in self.legs.values():
            if leg.symbol == symbol and leg.side == side:
                return leg
        return None
