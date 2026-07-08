"""Broker-level order manager with safe simulation defaults."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
from typing import Any, Protocol
from uuid import uuid4

from models import Mode, OrderSide, OrderStatus, OrderType
from models.order_models import OrderRequest, OrderResponse, TradeState
from utils.config_loader import ConfigLoader
from utils.exceptions import BrokerError, OrderValidationError
from utils.time_utils import now_ist

LOGGER = logging.getLogger("broker")
TRADE_LOGGER = logging.getLogger("trade")


class OrderExecutionAdapter(Protocol):
    """Protocol for real broker order execution calls."""

    def place_order(self, order_request: OrderRequest) -> dict[str, Any]:
        """Place an order and return broker response data."""

    def modify_order(
        self,
        order_id: str,
        new_price: float,
        new_trigger_price: float | None = None,
    ) -> dict[str, Any]:
        """Modify an existing order and return broker response data."""

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an existing order and return broker response data."""


class OrderManager:
    """Validate, log, simulate, or route broker-level order intents."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        mode: Mode | str | None = None,
        execution_adapter: OrderExecutionAdapter | None = None,
    ) -> None:
        """Create an order manager."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.mode = Mode(str(mode).upper()) if mode is not None else self.config.settings.mode
        self.execution_adapter = execution_adapter
        self.orders: dict[str, OrderResponse] = {}
        self.positions: dict[str, TradeState] = {}
        LOGGER.info("Order manager initialized", extra={"mode": self.mode.value})

    @property
    def is_simulated_mode(self) -> bool:
        """Return true for paper and backtest simulations."""

        return self.mode in {Mode.PAPER, Mode.BACKTEST}

    def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Validate and place or simulate an order."""

        self._log_order_intent("place_order", order_request)
        self._validate_order_request(order_request)
        if self.is_simulated_mode:
            return self._simulate_place_order(order_request)
        self._ensure_live_orders_allowed()
        if self.execution_adapter is None:
            raise BrokerError("Live order execution adapter is not configured")
        try:
            response = self.execution_adapter.place_order(order_request)
        except Exception as exc:
            LOGGER.error("Live place_order failed", extra={"symbol": order_request.symbol})
            raise BrokerError("Live place_order failed") from exc
        order_response = self._response_from_mapping(response, order_request)
        self.orders[order_response.order_id] = order_response
        return order_response

    def modify_order(
        self,
        order_id: str,
        new_price: float,
        new_trigger_price: float | None = None,
    ) -> OrderResponse:
        """Modify a live or simulated order."""

        self._log_order_intent(
            "modify_order",
            {"order_id": order_id, "new_price": new_price, "new_trigger_price": new_trigger_price},
        )
        self._validate_order_id(order_id)
        self._validate_positive_price(new_price, "new_price")
        if new_trigger_price is not None:
            self._validate_positive_price(new_trigger_price, "new_trigger_price")
        if self.is_simulated_mode:
            return self._simulate_modify_order(order_id, new_price, new_trigger_price)
        self._ensure_live_orders_allowed()
        if self.execution_adapter is None:
            raise BrokerError("Live order execution adapter is not configured")
        try:
            response = self.execution_adapter.modify_order(order_id, new_price, new_trigger_price)
        except Exception as exc:
            LOGGER.error("Live modify_order failed", extra={"order_id": order_id})
            raise BrokerError("Live modify_order failed") from exc
        original = self.orders.get(order_id)
        order_response = self._response_from_mapping(response, original)
        self.orders[order_response.order_id] = order_response
        return order_response

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel a live or simulated order."""

        self._log_order_intent("cancel_order", {"order_id": order_id})
        self._validate_order_id(order_id)
        if self.is_simulated_mode:
            return self._simulate_cancel_order(order_id)
        self._ensure_live_orders_allowed()
        if self.execution_adapter is None:
            raise BrokerError("Live order execution adapter is not configured")
        try:
            response = self.execution_adapter.cancel_order(order_id)
        except Exception as exc:
            LOGGER.error("Live cancel_order failed", extra={"order_id": order_id})
            raise BrokerError("Live cancel_order failed") from exc
        original = self.orders.get(order_id)
        order_response = self._response_from_mapping(response, original)
        self.orders[order_response.order_id] = order_response
        return order_response

    def cancel_all_orders(self) -> tuple[OrderResponse, ...]:
        """Cancel every open simulated or tracked order."""

        self._log_order_intent("cancel_all_orders", {"open_orders": len(self.orders)})
        cancellable = [
            order_id
            for order_id, order in self.orders.items()
            if order.status not in {OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED}
        ]
        return tuple(self.cancel_order(order_id) for order_id in cancellable)

    def squareoff_position(self, position: TradeState | dict[str, Any]) -> OrderResponse:
        """Square off a single position with a safe non-market order."""

        position_data = self._position_to_mapping(position)
        self._log_order_intent("squareoff_position", position_data)
        symbol = str(position_data.get("symbol") or "")
        quantity = int(position_data.get("quantity") or 0)
        side = OrderSide(str(position_data.get("side") or OrderSide.BUY).upper())
        if not symbol or quantity <= 0:
            raise OrderValidationError("Position must include symbol and positive quantity")
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        request = OrderRequest(
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            timestamp=now_ist(),
            limit_price=self._position_exit_price(position_data),
            tag="squareoff",
            metadata={"squareoff": True},
        )
        response = self.place_order(request)
        self.positions.pop(str(position_data.get("trade_id") or symbol), None)
        return response

    def squareoff_all(self) -> tuple[OrderResponse, ...]:
        """Square off every tracked simulated position."""

        self._log_order_intent("squareoff_all", {"positions": len(self.positions)})
        return tuple(self.squareoff_position(position) for position in tuple(self.positions.values()))

    def _ensure_live_orders_allowed(self) -> None:
        if self.mode != Mode.LIVE or not self.config.settings.live_trading_enabled:
            raise BrokerError("LIVE orders require mode=LIVE and live_trading_enabled=true")

    def _validate_order_request(self, order_request: OrderRequest) -> None:
        if not isinstance(order_request, OrderRequest):
            raise OrderValidationError("order_request must be an OrderRequest")
        if not order_request.symbol:
            raise OrderValidationError("Order symbol is required")
        if order_request.quantity <= 0:
            raise OrderValidationError("Order quantity must be positive")
        if not isinstance(order_request.side, OrderSide):
            raise OrderValidationError("Order side must be an OrderSide")
        if not isinstance(order_request.order_type, OrderType):
            raise OrderValidationError("Order type must be an OrderType")
        if order_request.order_type == OrderType.LIMIT:
            self._validate_positive_price(order_request.limit_price, "limit_price")
        if order_request.order_type == OrderType.SL_LIMIT:
            self._validate_positive_price(order_request.limit_price, "limit_price")
            self._validate_positive_price(order_request.trigger_price, "trigger_price")
        if order_request.order_type == OrderType.MARKET:
            self._validate_market_order_exception(order_request)

    def _validate_market_order_exception(self, order_request: OrderRequest) -> None:
        if not order_request.is_emergency_exit:
            raise OrderValidationError("Market orders are allowed only for emergency exits")
        if not self.config.risk_limits.emergency_market_exit_allowed:
            raise OrderValidationError("Emergency market exits are disabled by risk config")

    @staticmethod
    def _validate_order_id(order_id: str) -> None:
        if not isinstance(order_id, str) or not order_id.strip():
            raise OrderValidationError("order_id is required")

    @staticmethod
    def _validate_positive_price(value: float | None, name: str) -> None:
        if not isinstance(value, int | float) or value <= 0:
            raise OrderValidationError(f"{name} must be a positive number")

    def _simulate_place_order(self, order_request: OrderRequest) -> OrderResponse:
        order_id = f"{self.mode.value.lower()}-{uuid4().hex[:12]}"
        status = OrderStatus.FILLED if order_request.order_type == OrderType.MARKET else OrderStatus.OPEN
        average_price = self._simulated_average_price(order_request)
        response = OrderResponse(
            order_id=order_id,
            status=status,
            timestamp=now_ist(),
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            filled_quantity=order_request.quantity if status == OrderStatus.FILLED else 0,
            average_price=average_price if status == OrderStatus.FILLED else None,
            raw_response={"mode": self.mode.value, "simulated": True},
        )
        self.orders[order_id] = response
        if status == OrderStatus.FILLED:
            self._track_filled_position(order_request, response)
        LOGGER.info("Simulated order accepted", extra={"order_id": order_id, "status": status.value})
        return response

    def _simulate_modify_order(
        self,
        order_id: str,
        new_price: float,
        new_trigger_price: float | None,
    ) -> OrderResponse:
        original = self.orders.get(order_id)
        if original is None:
            raise BrokerError(f"Order not found: {order_id}")
        if original.status in {OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED}:
            raise BrokerError(f"Order cannot be modified in status {original.status.value}")
        response = OrderResponse(
            order_id=order_id,
            status=OrderStatus.OPEN,
            timestamp=now_ist(),
            symbol=original.symbol,
            side=original.side,
            quantity=original.quantity,
            filled_quantity=original.filled_quantity,
            average_price=original.average_price,
            raw_response={
                **original.raw_response,
                "modified": True,
                "new_price": new_price,
                "new_trigger_price": new_trigger_price,
            },
        )
        self.orders[order_id] = response
        return response

    def _simulate_cancel_order(self, order_id: str) -> OrderResponse:
        original = self.orders.get(order_id)
        if original is None:
            raise BrokerError(f"Order not found: {order_id}")
        response = OrderResponse(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            timestamp=now_ist(),
            symbol=original.symbol,
            side=original.side,
            quantity=original.quantity,
            filled_quantity=original.filled_quantity,
            average_price=original.average_price,
            raw_response={**original.raw_response, "cancelled": True},
        )
        self.orders[order_id] = response
        return response

    def _track_filled_position(self, request: OrderRequest, response: OrderResponse) -> None:
        self.positions[response.order_id] = TradeState(
            trade_id=response.order_id,
            symbol=response.symbol,
            side=response.side,
            quantity=response.quantity,
            entry_time=response.timestamp,
            entry_price=response.average_price or self._simulated_average_price(request),
            status=OrderStatus.OPEN,
            strategy_id=request.strategy_id,
            metadata={"mode": self.mode.value, "simulated": True},
        )

    @staticmethod
    def _simulated_average_price(request: OrderRequest) -> float:
        if request.limit_price is not None:
            return float(request.limit_price)
        if request.trigger_price is not None:
            return float(request.trigger_price)
        return 0.0

    @staticmethod
    def _position_to_mapping(position: TradeState | dict[str, Any]) -> dict[str, Any]:
        if is_dataclass(position) and not isinstance(position, type):
            return asdict(position)
        if isinstance(position, dict):
            return dict(position)
        raise OrderValidationError("position must be a TradeState or mapping")

    @staticmethod
    def _position_exit_price(position_data: dict[str, Any]) -> float:
        for key in ("exit_price", "last_price", "entry_price", "average_price"):
            value = position_data.get(key)
            if isinstance(value, int | float) and value > 0:
                return float(value)
        raise OrderValidationError("Position must include a positive exit, last, entry, or average price")

    def _response_from_mapping(
        self,
        response: dict[str, Any],
        fallback: OrderRequest | OrderResponse | None,
    ) -> OrderResponse:
        if not isinstance(response, dict):
            raise BrokerError("Broker order response must be a mapping")
        fallback_order_id = fallback.order_id if isinstance(fallback, OrderResponse) else None
        fallback_symbol = fallback.symbol if fallback is not None else ""
        fallback_side = fallback.side if fallback is not None else OrderSide.BUY
        fallback_quantity = fallback.quantity if fallback is not None else 0
        order_id = str(response.get("order_id") or response.get("id") or fallback_order_id or "")
        if not order_id:
            raise BrokerError("Broker order response missing order_id")
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus(str(response.get("status") or OrderStatus.SUBMITTED).upper()),
            timestamp=now_ist(),
            symbol=str(response.get("symbol") or fallback_symbol),
            side=OrderSide(str(response.get("side") or fallback_side).upper()),
            quantity=int(response.get("quantity") or fallback_quantity),
            filled_quantity=int(response.get("filled_quantity") or response.get("filled_qty") or 0),
            average_price=self._optional_float(response.get("average_price") or response.get("avg_price")),
            rejection_reason=self._optional_str(response.get("rejection_reason")),
            raw_response=dict(response),
        )

    def _log_order_intent(self, action: str, payload: Any) -> None:
        safe_payload = self._safe_payload(payload)
        LOGGER.info("Order intent", extra={"action": action, "payload": safe_payload})
        TRADE_LOGGER.info("Order intent", extra={"action": action, "payload": safe_payload})

    @staticmethod
    def _safe_payload(payload: Any) -> dict[str, Any]:
        if is_dataclass(payload) and not isinstance(payload, type):
            data = asdict(payload)
        elif isinstance(payload, dict):
            data = dict(payload)
        else:
            data = {"value": str(payload)}
        data.pop("metadata", None)
        return data

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return None if value is None else str(value)
