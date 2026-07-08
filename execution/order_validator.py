"""Order validation for safe execution pricing and order-type policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

from execution.order_pricing import convert_to_emergency_market_order, parse_quote
from models import OrderSide, OrderType
from models.order_models import OrderRequest
from utils.config_loader import ConfigLoader
from utils.exceptions import OrderValidationError
from utils.math_utils import safe_divide
from utils.time_utils import ensure_ist, now_ist

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderPriceValidationResult:
    """Structured order-price validation result."""

    approved: bool
    reason: str
    bid_ask_spread_percent: float
    slippage_percent: float
    warnings: tuple[str, ...] = ()


class OrderValidator:
    """Validate order prices, order types, timeouts, and emergency conversion."""

    def __init__(self, *, config_loader: ConfigLoader | None = None) -> None:
        """Create order validator from app config."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()

    def validate_order_price(self, order_request: OrderRequest, quote: Any) -> OrderPriceValidationResult:
        """Validate order type and price against quote and risk limits."""

        return validate_order_price(order_request, quote, config_loader=self.config_loader)

    def should_cancel_unfilled(
        self,
        order_request: OrderRequest,
        *,
        submitted_at: datetime,
        now: datetime | None = None,
    ) -> bool:
        """Return true when an unfilled order should be cancelled after timeout."""

        timeout = self.config.risk_limits.order_timeout_seconds
        elapsed = (ensure_ist(now or now_ist()) - ensure_ist(submitted_at)).total_seconds()
        return elapsed >= timeout and order_request.order_type != OrderType.MARKET

    def emergency_market_exit(
        self,
        order_request: OrderRequest,
        *,
        submitted_at: datetime,
        now: datetime | None = None,
        reason: str = "SL-LIMIT not filled within timeout",
    ) -> OrderRequest | None:
        """Return emergency MARKET conversion when allowed and timed out."""

        if not self.should_cancel_unfilled(order_request, submitted_at=submitted_at, now=now):
            return None
        if order_request.order_type != OrderType.SL_LIMIT or not order_request.metadata.get("short_option_exit"):
            return None
        if not self.config.risk_limits.emergency_market_exit_allowed:
            return None
        LOGGER.warning("Emergency market exit allowed", extra={"symbol": order_request.symbol, "reason": reason})
        return convert_to_emergency_market_order(order_request, reason=reason)


def validate_order_price(
    order_request: OrderRequest,
    quote: Any,
    *,
    config_loader: ConfigLoader | None = None,
) -> OrderPriceValidationResult:
    """Validate order request price and quote constraints."""

    if not isinstance(order_request, OrderRequest):
        raise OrderValidationError("order_request must be an OrderRequest")
    config = (config_loader or ConfigLoader()).load()
    parsed_quote = parse_quote(quote)
    _validate_basic_order(order_request)
    _validate_order_type_policy(order_request, config.risk_limits.emergency_market_exit_allowed)
    bid_ask_spread_percent = _bid_ask_spread_percent(parsed_quote.bid, parsed_quote.ask)
    if bid_ask_spread_percent > config.risk_limits.max_bid_ask_spread_percent:
        raise OrderValidationError("Bid-ask spread limit exceeded")
    slippage_percent = _estimated_slippage_percent(order_request, parsed_quote.ltp)
    if slippage_percent > config.risk_limits.max_slippage_percent:
        raise OrderValidationError("Slippage limit exceeded")
    warnings: list[str] = []
    if bid_ask_spread_percent > config.risk_limits.max_bid_ask_spread_percent * 0.8:
        warnings.append("bid_ask_spread_near_limit")
    if slippage_percent > config.risk_limits.max_slippage_percent * 0.8:
        warnings.append("slippage_near_limit")
    return OrderPriceValidationResult(
        approved=True,
        reason="Order price validation passed.",
        bid_ask_spread_percent=bid_ask_spread_percent,
        slippage_percent=slippage_percent,
        warnings=tuple(warnings),
    )


def _validate_basic_order(order_request: OrderRequest) -> None:
    if not order_request.symbol:
        raise OrderValidationError("Order symbol is required")
    if order_request.quantity <= 0:
        raise OrderValidationError("Order quantity must be positive")
    if order_request.order_type == OrderType.LIMIT and (order_request.limit_price is None or order_request.limit_price <= 0):
        raise OrderValidationError("LIMIT order requires a positive limit_price")
    if order_request.order_type == OrderType.SL_LIMIT:
        if order_request.trigger_price is None or order_request.trigger_price <= 0:
            raise OrderValidationError("SL_LIMIT order requires a positive trigger_price")
        if order_request.limit_price is None or order_request.limit_price <= 0:
            raise OrderValidationError("SL_LIMIT order requires a positive limit_price")


def _validate_order_type_policy(order_request: OrderRequest, emergency_market_exit_allowed: bool) -> None:
    is_option = _is_option_order(order_request)
    is_short_entry = order_request.side == OrderSide.SELL and not order_request.metadata.get("exit_reason")
    is_short_exit = order_request.side == OrderSide.BUY and bool(order_request.metadata.get("short_option_exit"))

    if order_request.order_type == OrderType.MARKET:
        if not order_request.is_emergency_exit:
            raise OrderValidationError("MARKET orders are allowed only for emergency exits")
        if not emergency_market_exit_allowed:
            raise OrderValidationError("Emergency market exits are disabled by risk config")
        return

    if is_option and is_short_entry and order_request.order_type != OrderType.LIMIT:
        raise OrderValidationError("Option selling entries must use LIMIT orders")
    if is_short_exit and order_request.metadata.get("exit_reason") == "profit_booking" and order_request.order_type != OrderType.LIMIT:
        raise OrderValidationError("Short option profit booking must use LIMIT BUY")
    if is_short_exit and order_request.metadata.get("exit_reason") == "stop_loss" and order_request.order_type != OrderType.SL_LIMIT:
        raise OrderValidationError("Short option stop-loss exits must use BUY SL-LIMIT")


def _is_option_order(order_request: OrderRequest) -> bool:
    option_type = str(order_request.metadata.get("option_type", "")).upper()
    symbol = order_request.symbol.upper()
    return option_type in {"CE", "PE"} or symbol.endswith("CE") or symbol.endswith("PE")


def _bid_ask_spread_percent(bid: float | None, ask: float | None) -> float:
    if bid is None or ask is None:
        return 0.0
    mid = (bid + ask) / 2.0
    return safe_divide(ask - bid, mid, 0.0) * 100.0


def _estimated_slippage_percent(order_request: OrderRequest, ltp: float) -> float:
    if order_request.order_type == OrderType.MARKET:
        return 0.0
    reference = order_request.limit_price
    if reference is None:
        return 0.0
    return abs(reference - ltp) / ltp * 100.0
