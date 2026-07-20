"""Order routing for execution baskets."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable

from broker.order_manager import OrderManager
from execution.order_validator import OrderValidator
from execution.retry_handler import RetryHandler
from models import Mode, OrderResponse, OrderStatus, OrderType, StrategyLeg, StrategyPlan
from models.order_models import OrderRequest
from utils.config_loader import ConfigLoader
from utils.exceptions import ExecutionError, OrderValidationError
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)
QuoteProvider = Callable[[str], Any]
LiveSafetyApproval = Callable[[], object]


@dataclass(frozen=True)
class OrderBasket:
    """Prepared basket of order requests."""

    strategy_id: str
    orders: tuple[OrderRequest, ...]


class OrderRouter:
    """Prepare, validate, place, and cancel order baskets."""

    def __init__(
        self,
        *,
        order_manager: OrderManager | None = None,
        order_validator: OrderValidator | None = None,
        config_loader: ConfigLoader | None = None,
        retry_handler: RetryHandler | None = None,
        live_safety_approval: LiveSafetyApproval | None = None,
    ) -> None:
        """Create order router."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.order_manager = order_manager or OrderManager(config_loader=self.config_loader)
        self.order_validator = order_validator or OrderValidator(config_loader=self.config_loader)
        self.retry_handler = retry_handler or RetryHandler()
        self.live_safety_approval = live_safety_approval

    def prepare_entry_basket(self, plan: StrategyPlan) -> OrderBasket:
        """Convert StrategyPlan legs into entry LIMIT order requests."""

        orders = tuple(self._order_from_leg(plan, leg) for leg in plan.legs)
        if not orders:
            raise ExecutionError("StrategyPlan has no executable legs")
        return OrderBasket(strategy_id=plan.strategy_id, orders=orders)

    def validate_basket(self, basket: OrderBasket, quote_provider: QuoteProvider) -> None:
        """Validate each order in a basket against latest quote."""

        for order in basket.orders:
            self.order_validator.validate_order_price(order, quote_provider(order.symbol))

    def place_basket(self, basket: OrderBasket, quote_provider: QuoteProvider) -> tuple[OrderResponse, ...]:
        """Validate and place all orders in a basket."""

        self._ensure_live_allowed_if_needed()
        responses: list[OrderResponse] = []
        for order in basket.orders:
            self.order_validator.validate_order_price(order, quote_provider(order.symbol))
            response = self.retry_handler.run(
                lambda order=order: self.order_manager.place_order(order),
                operation_name=f"place_order:{order.symbol}",
            )
            responses.append(response)
            if response.status in {OrderStatus.FAILED}:
                raise ExecutionError("Dependent basket legs blocked by unresolved prior order")
        return tuple(responses)

    def place_order(self, order_request: OrderRequest, quote_provider: QuoteProvider) -> OrderResponse:
        """Validate and place a single order."""

        self._ensure_live_allowed_if_needed()
        self.order_validator.validate_order_price(order_request, quote_provider(order_request.symbol))
        return self.retry_handler.run(
            lambda: self.order_manager.place_order(order_request),
            operation_name=f"place_order:{order_request.symbol}",
        )

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel one routed order."""

        return self.retry_handler.run(
            lambda: self.order_manager.cancel_order(order_id),
            operation_name=f"cancel_order:{order_id}",
        )

    def cancel_unfilled(self, responses: tuple[OrderResponse, ...]) -> tuple[OrderResponse, ...]:
        """Cancel all non-filled orders."""

        cancelled: list[OrderResponse] = []
        for response in responses:
            if response.status not in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                cancelled.append(self.cancel_order(response.order_id))
        return tuple(cancelled)

    def _ensure_live_allowed_if_needed(self) -> None:
        mode = self.config.settings.mode
        if mode != Mode.LIVE:
            return
        if not (
            self.config.settings.live_trading_enabled
            and self.config.settings.live_order_submission_enabled
        ):
            raise ExecutionError("LIVE order submission switches are not enabled")
        if self.live_safety_approval is None:
            raise ExecutionError("Live safety approval is required before routing")
        self.live_safety_approval()

    def _order_from_leg(self, plan: StrategyPlan, leg: StrategyLeg) -> OrderRequest:
        if leg.order_type == OrderType.MARKET and not bool(leg.metadata.get("emergency_exit")):
            raise OrderValidationError("Entry MARKET orders are not allowed")
        order_type = leg.order_type if leg.order_type != OrderType.MARKET else OrderType.MARKET
        if not leg.metadata.get("emergency_exit"):
            order_type = OrderType.LIMIT
        return OrderRequest(
            symbol=leg.symbol,
            side=leg.side,
            quantity=leg.quantity,
            order_type=order_type,
            timestamp=now_ist(),
            strategy_id=plan.strategy_id,
            limit_price=leg.limit_price,
            trigger_price=leg.trigger_price,
            product_type=self.config.broker.orders.get("product_type", "MIS"),
            validity=self.config.broker.orders.get("validity", "DAY"),
            tag="entry",
            metadata={
                **leg.metadata,
                "strategy_type": plan.strategy_type.value,
                "entry": True,
            },
        )
