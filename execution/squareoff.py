"""Squareoff helpers for execution recovery and manual exits."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Any

from execution.order_pricing import calculate_limit_price
from execution.order_router import OrderRouter
from execution.trade_state import ExecutionTradeState, LegExecutionState
from models import OrderResponse, OrderSide, OrderType
from models.order_models import OrderRequest
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)
QuoteProvider = Callable[[str], Any]


@dataclass(frozen=True)
class SquareoffResult:
    """Result of a squareoff action."""

    reason: str
    orders: tuple[OrderRequest, ...]
    responses: tuple[OrderResponse, ...]


class SquareoffManager:
    """Create and route squareoff orders for filled execution legs."""

    def __init__(self, *, order_router: OrderRouter) -> None:
        """Create squareoff manager."""

        self.order_router = order_router

    def squareoff_filled_legs(
        self,
        state: ExecutionTradeState,
        *,
        quote_provider: QuoteProvider,
        reason: str,
    ) -> SquareoffResult:
        """Square off every filled leg in the execution state."""

        orders = tuple(
            self._squareoff_order(leg, quote_provider(leg.symbol), state.strategy_id, reason)
            for leg in state.filled_legs()
        )
        responses = tuple(self.order_router.place_order(order, quote_provider) for order in orders)
        state.exit_reason = reason
        LOGGER.warning("Squareoff routed", extra={"strategy_id": state.strategy_id, "reason": reason, "orders": len(orders)})
        return SquareoffResult(reason=reason, orders=orders, responses=responses)

    def handle_partial_fill(
        self,
        state: ExecutionTradeState,
        *,
        quote_provider: QuoteProvider,
    ) -> SquareoffResult | None:
        """Square off partial fills to avoid unhedged exposure."""

        if not state.has_partial_fill():
            return None
        return self.squareoff_filled_legs(state, quote_provider=quote_provider, reason="partial_fill_squareoff")

    def manual_squareoff(
        self,
        state: ExecutionTradeState,
        *,
        quote_provider: QuoteProvider,
    ) -> SquareoffResult:
        """Square off because of manual operator request."""

        state.manual_squareoff_requested = True
        return self.squareoff_filled_legs(state, quote_provider=quote_provider, reason="manual_squareoff")

    @staticmethod
    def _squareoff_order(
        leg: LegExecutionState,
        quote: Any,
        strategy_id: str,
        reason: str,
    ) -> OrderRequest:
        quote_ltp, bid, ask = _quote_parts(quote)
        exit_side = OrderSide.SELL if leg.side == OrderSide.BUY else OrderSide.BUY
        return OrderRequest(
            symbol=leg.symbol,
            side=exit_side,
            quantity=max(leg.filled_quantity, 0),
            order_type=OrderType.LIMIT,
            timestamp=now_ist(),
            strategy_id=strategy_id,
            limit_price=calculate_limit_price(exit_side, quote_ltp, bid, ask, 0.0),
            tag="squareoff",
            metadata={
                "squareoff": True,
                "exit_reason": reason,
                "short_option_exit": exit_side == OrderSide.BUY,
            },
        )


def _quote_parts(quote: Any) -> tuple[float, float | None, float | None]:
    if isinstance(quote, dict):
        ltp = quote.get("ltp", quote.get("last_price"))
        return float(ltp), _optional_float(quote.get("bid")), _optional_float(quote.get("ask"))
    ltp = getattr(quote, "ltp", None)
    if ltp is None:
        ltp = getattr(quote, "last_price", None)
    return float(ltp), _optional_float(getattr(quote, "bid", None)), _optional_float(getattr(quote, "ask", None))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
