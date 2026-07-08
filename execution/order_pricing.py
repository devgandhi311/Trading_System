"""Order pricing helpers for safe option execution."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from models import OrderSide, OrderType
from models.order_models import OrderRequest
from utils.exceptions import ExecutionError
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceQuote:
    """Minimal quote used by execution pricing."""

    ltp: float
    bid: float | None = None
    ask: float | None = None


def calculate_limit_price(
    side: OrderSide | str,
    ltp: float,
    bid: float | None,
    ask: float | None,
    buffer: float = 0.0,
) -> float:
    """Calculate a conservative limit price near best bid/ask."""

    normalized_side = OrderSide(str(side).upper())
    if normalized_side == OrderSide.BUY:
        reference = ask if ask is not None and ask > 0 else ltp
        return _positive_price(reference + abs(buffer), "limit_price")
    reference = bid if bid is not None and bid > 0 else ltp
    return _positive_price(reference - abs(buffer), "limit_price")


def calculate_trigger_price(
    entry_price: float,
    stop_loss_percent: float,
    side: OrderSide | str,
) -> float:
    """Calculate stop-loss trigger from entry price and side."""

    entry = _positive_price(entry_price, "entry_price")
    percent = _non_negative(stop_loss_percent, "stop_loss_percent") / 100.0
    normalized_side = OrderSide(str(side).upper())
    if normalized_side == OrderSide.SELL:
        return entry * (1.0 + percent)
    return entry * (1.0 - percent)


def calculate_sl_limit_price(
    trigger_price: float,
    side: OrderSide | str,
    buffer: float = 0.0,
) -> float:
    """Calculate SL-LIMIT limit price from trigger and side."""

    trigger = _positive_price(trigger_price, "trigger_price")
    normalized_side = OrderSide(str(side).upper())
    if normalized_side == OrderSide.BUY:
        return _positive_price(trigger + abs(buffer), "limit_price")
    return _positive_price(trigger - abs(buffer), "limit_price")


def build_profit_booking_order(
    *,
    symbol: str,
    quantity: int,
    quote: Any,
    buffer: float = 0.0,
    strategy_id: str | None = None,
) -> OrderRequest:
    """Build a LIMIT BUY order for short-option profit booking."""

    parsed = parse_quote(quote)
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        timestamp=now_ist(),
        strategy_id=strategy_id,
        limit_price=calculate_limit_price(OrderSide.BUY, parsed.ltp, parsed.bid, parsed.ask, buffer),
        tag="profit_booking",
        metadata={"exit_reason": "profit_booking", "short_option_exit": True},
    )


def build_short_option_stop_loss_order(
    *,
    symbol: str,
    quantity: int,
    entry_price: float,
    stop_loss_percent: float,
    buffer: float = 1.0,
    strategy_id: str | None = None,
) -> OrderRequest:
    """Build BUY SL-LIMIT stop-loss order for short option exit."""

    trigger = calculate_trigger_price(entry_price, stop_loss_percent, OrderSide.SELL)
    limit = calculate_sl_limit_price(trigger, OrderSide.BUY, buffer)
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.SL_LIMIT,
        timestamp=now_ist(),
        strategy_id=strategy_id,
        limit_price=limit,
        trigger_price=trigger,
        tag="stop_loss",
        metadata={"exit_reason": "stop_loss", "short_option_exit": True},
    )


def convert_to_emergency_market_order(order_request: OrderRequest, *, reason: str) -> OrderRequest:
    """Convert an unfilled exit order into an emergency MARKET order."""

    if order_request.side != OrderSide.BUY:
        raise ExecutionError("Emergency market conversion is supported only for BUY exits")
    LOGGER.warning(
        "Converting exit order to emergency MARKET",
        extra={"symbol": order_request.symbol, "reason": reason},
    )
    return OrderRequest(
        symbol=order_request.symbol,
        side=order_request.side,
        quantity=order_request.quantity,
        order_type=OrderType.MARKET,
        timestamp=now_ist(),
        strategy_id=order_request.strategy_id,
        product_type=order_request.product_type,
        validity=order_request.validity,
        tag=order_request.tag,
        metadata={**order_request.metadata, "emergency_exit": True, "emergency_reason": reason},
    )


def parse_quote(quote: Any) -> PriceQuote:
    """Parse a mapping or QuoteSnapshot-like object into PriceQuote."""

    if isinstance(quote, dict):
        ltp = quote.get("ltp", quote.get("last_price"))
        bid = quote.get("bid")
        ask = quote.get("ask")
    else:
        ltp = getattr(quote, "ltp", None)
        if ltp is None:
            ltp = getattr(quote, "last_price", None)
        bid = getattr(quote, "bid", None)
        ask = getattr(quote, "ask", None)
    return PriceQuote(
        ltp=_positive_price(float(ltp), "ltp"),
        bid=_optional_positive(bid, "bid"),
        ask=_optional_positive(ask, "ask"),
    )


def _positive_price(value: float, name: str) -> float:
    if value <= 0:
        raise ExecutionError(f"{name} must be positive")
    return float(value)


def _optional_positive(value: Any, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ExecutionError(f"{name} must be positive when supplied")
    return parsed


def _non_negative(value: float, name: str) -> float:
    if value < 0:
        raise ExecutionError(f"{name} must be non-negative")
    return float(value)
