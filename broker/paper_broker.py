"""Authoritative quote-driven PAPER broker with transactional persistence."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import logging
from threading import RLock
from typing import Callable

from backtesting.transaction_costs import TransactionCostModel
from models.common import OrderSide, OrderStatus, OrderType
from models.order_models import OrderRequest, OrderResponse
from models.paper_models import MarketSnapshot, PaperFill, PaperOrder
from storage.repositories import OperationalRepository
from utils.time_utils import ensure_ist, now_ist

LOGGER = logging.getLogger(__name__)
Clock = Callable[[], datetime]
TERMINAL_STATES = {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}


class PaperBrokerError(RuntimeError):
    """Raised when a PAPER operation fails closed."""


class PaperBroker:
    """Persist and simulate PAPER orders without any live broker adapter."""

    def __init__(
        self,
        repository: OperationalRepository,
        *,
        session_id: str,
        clock: Clock = now_ist,
        transaction_cost_model: TransactionCostModel | None = None,
        max_quote_age_seconds: float = 5.0,
        latency: timedelta = timedelta(0),
        slippage_percent: float = 0.0,
        emergency_slippage_percent: float = 0.2,
        emergency_market_exit_allowed: bool = True,
        order_timeout: timedelta = timedelta(seconds=30),
    ) -> None:
        self.repository = repository
        self.session_id = session_id
        self.clock = clock
        self.cost_model = transaction_cost_model or TransactionCostModel()
        self.max_quote_age_seconds = max_quote_age_seconds
        self.latency = latency
        self.slippage_percent = slippage_percent
        self.emergency_slippage_percent = emergency_slippage_percent
        self.emergency_market_exit_allowed = emergency_market_exit_allowed
        self.order_timeout = order_timeout
        self._lock = RLock()

    def submit_order(
        self,
        request: OrderRequest,
        quote: MarketSnapshot | None = None,
        *,
        instrument_token: str | None = None,
        lot_size: int = 1,
    ) -> PaperOrder:
        """Reserve, validate, submit, and optionally process one PAPER order."""

        if request.quantity <= 0:
            raise PaperBrokerError("quantity must be positive")
        key = request.idempotency_key
        token = instrument_token or str(request.metadata.get("instrument_token") or "")
        if not token:
            raise PaperBrokerError("instrument_token is required")
        if lot_size <= 0:
            raise PaperBrokerError("lot_size must be positive")
        with self._lock:
            existing = self.repository.get_order_intent(key)
            if existing is not None:
                return self.get_order(key)
            metadata = {
                "paper": True,
                "instrument_token": token,
                "lot_size": lot_size,
                "strategy_id": request.strategy_id or "paper-strategy",
                "leg_id": str(request.metadata.get("leg_id") or "paper-leg"),
                "request_timestamp": ensure_ist(request.timestamp).isoformat(),
                "submitted_at": ensure_ist(self.clock()).isoformat(),
                "remaining_quantity": request.quantity,
                "average_fill_price": None,
                "entry_turnover": 0.0,
                "transaction_costs": 0.0,
                "processed_quote_ids": [],
                "modification_history": [],
                "triggered": False,
            }
            row = self.repository.reserve_order_intent(
                idempotency_key=key, client_order_id=key, session_id=self.session_id,
                strategy_id=request.strategy_id, leg_id=metadata["leg_id"], instrument_token=token,
                symbol=request.symbol, side=request.side.value, order_type=request.order_type.value,
                quantity=request.quantity, limit_price=request.limit_price, trigger_price=request.trigger_price,
                product_type=request.product_type, validity=request.validity, local_state="CREATED",
                created_at=ensure_ist(self.clock()), metadata=metadata,
            )
            rejection = self._request_rejection(request)
            if rejection:
                metadata["rejection_reason"] = rejection
                self.repository.apply_paper_order_update(key, "REJECTED", expected_version=row.version,
                                                         metadata=metadata, detail={"reason": rejection})
                return self.get_order(key)
            for state in ("VALIDATED", "RESERVED", "SUBMITTED"):
                row = self.repository.apply_paper_order_update(
                    key, state, expected_version=row.version, metadata=metadata, detail={"paper": True}
                )
            initial = "TRIGGER_PENDING" if request.order_type == OrderType.SL_LIMIT else "OPEN"
            row = self.repository.apply_paper_order_update(
                key, initial, expected_version=row.version, metadata=metadata, detail={"paper": True}
            )
            if quote is not None:
                return self.process_quote(key, quote)
            return self.get_order(key)

    def process_quote(self, idempotency_key: str, quote: MarketSnapshot) -> PaperOrder:
        """Apply a quote once and atomically persist any resulting fill."""

        with self._lock:
            row = self._required_row(idempotency_key)
            if row.local_state in TERMINAL_STATES:
                return self.get_order(idempotency_key)
            metadata = dict(row.metadata_json)
            processed = list(metadata.get("processed_quote_ids", []))
            if quote.snapshot_id in processed:
                return self.get_order(idempotency_key)
            now = ensure_ist(self.clock())
            quote_time = ensure_ist(quote.timestamp)
            reason = self._quote_rejection(quote, now)
            if reason:
                processed.append(quote.snapshot_id)
                metadata.update({"processed_quote_ids": processed, "rejection_reason": reason})
                self.repository.apply_paper_order_update(
                    idempotency_key, "REJECTED", expected_version=row.version,
                    metadata=metadata, detail={"quote_id": quote.snapshot_id, "reason": reason}, occurred_at=now,
                )
                return self.get_order(idempotency_key)
            submitted_at = ensure_ist(datetime.fromisoformat(str(metadata["submitted_at"])))
            if now - submitted_at > self.order_timeout:
                metadata["rejection_reason"] = "order timeout"
                self.repository.apply_paper_order_update(
                    idempotency_key, "EXPIRED", expected_version=row.version, metadata=metadata,
                    detail={"quote_id": quote.snapshot_id, "reason": "order timeout"}, occurred_at=now,
                )
                return self.get_order(idempotency_key)
            if now < submitted_at + self.latency:
                return self.get_order(idempotency_key)
            if row.local_state == "TRIGGER_PENDING" and not metadata.get("triggered"):
                triggered = quote.ask >= float(row.trigger_price) if row.side == "BUY" else quote.bid <= float(row.trigger_price)
                if not triggered:
                    processed.append(quote.snapshot_id)
                    metadata["processed_quote_ids"] = processed
                    self.repository.apply_paper_order_update(
                        idempotency_key, "TRIGGER_PENDING", expected_version=row.version,
                        metadata=metadata, detail={"quote_id": quote.snapshot_id, "triggered": False}, occurred_at=now,
                    )
                    return self.get_order(idempotency_key)
                metadata["triggered"] = True
                row = self.repository.apply_paper_order_update(
                    idempotency_key, "OPEN", expected_version=row.version, metadata=metadata,
                    detail={"quote_id": quote.snapshot_id, "triggered": True}, occurred_at=now,
                )
            price = self._executable_price(row, quote)
            processed.append(quote.snapshot_id)
            metadata["processed_quote_ids"] = processed
            if price is None:
                self.repository.apply_paper_order_update(
                    idempotency_key, "OPEN", expected_version=row.version, metadata=metadata,
                    detail={"quote_id": quote.snapshot_id, "marketable": False}, occurred_at=now,
                )
                return self.get_order(idempotency_key)
            remaining = row.quantity - row.filled_quantity
            fill_quantity = min(remaining, quote.available_liquidity)
            fill_price = self._slipped_price(price, row.side, row.order_type == OrderType.MARKET.value)
            if row.order_type != OrderType.MARKET.value:
                fill_price = min(fill_price, float(row.limit_price)) if row.side == "BUY" else max(
                    fill_price, float(row.limit_price)
                )
            old_turnover = float(metadata.get("entry_turnover", 0.0))
            turnover = fill_price * fill_quantity
            new_filled = row.filled_quantity + fill_quantity
            average = (old_turnover + turnover) / new_filled
            cost = self.cost_model.estimate(price=fill_price, quantity=fill_quantity)
            metadata.update(
                remaining_quantity=row.quantity - new_filled,
                average_fill_price=average,
                entry_turnover=old_turnover + turnover,
                transaction_costs=float(metadata.get("transaction_costs", 0.0)) + cost,
            )
            fill_id = hashlib.sha256(
                f"{idempotency_key}|{quote.snapshot_id}|{row.filled_quantity}|{fill_quantity}".encode("utf-8")
            ).hexdigest()
            state = "FILLED" if new_filled == row.quantity else "PARTIALLY_FILLED"
            self.repository.apply_fill_and_update_order(
                fill_id=fill_id, idempotency_key=idempotency_key, quantity=fill_quantity,
                price=fill_price, transaction_cost=cost, new_state=state,
                expected_version=row.version, metadata=metadata, filled_at=now,
            )
            return self.get_order(idempotency_key)

    def cancel_order(self, idempotency_key: str, *, reason: str = "operator cancellation") -> PaperOrder:
        """Cancel an open order without altering persisted fills."""

        with self._lock:
            row = self._required_row(idempotency_key)
            if row.local_state in TERMINAL_STATES:
                raise PaperBrokerError(f"cannot cancel order in {row.local_state}")
            metadata = dict(row.metadata_json)
            metadata["cancellation_reason"] = reason
            row = self.repository.apply_paper_order_update(
                idempotency_key, "CANCEL_PENDING", expected_version=row.version, metadata=metadata,
                detail={"reason": reason},
            )
            self.repository.apply_paper_order_update(
                idempotency_key, "CANCELLED", expected_version=row.version, metadata=metadata,
                detail={"reason": reason},
            )
            return self.get_order(idempotency_key)

    def modify_order(
        self, idempotency_key: str, *, limit_price: float, trigger_price: float | None = None
    ) -> PaperOrder:
        """Persist a valid price modification while retaining prior events and fills."""

        if limit_price <= 0 or (trigger_price is not None and trigger_price <= 0):
            raise PaperBrokerError("modified prices must be positive")
        with self._lock:
            row = self._required_row(idempotency_key)
            if row.local_state in TERMINAL_STATES:
                raise PaperBrokerError(f"cannot modify order in {row.local_state}")
            if row.order_type == OrderType.SL_LIMIT.value and trigger_price is None:
                raise PaperBrokerError("SL-LIMIT modification requires trigger_price")
            metadata = dict(row.metadata_json)
            history = list(metadata.get("modification_history", []))
            history.append({"limit_price": limit_price, "trigger_price": trigger_price,
                            "modified_at": ensure_ist(self.clock()).isoformat()})
            metadata["modification_history"] = history
            row = self.repository.apply_paper_order_update(
                idempotency_key, "MODIFY_PENDING", expected_version=row.version, metadata=metadata,
                detail=history[-1], limit_price=limit_price, trigger_price=trigger_price,
            )
            destination = "TRIGGER_PENDING" if row.order_type == OrderType.SL_LIMIT.value and not metadata.get("triggered") else (
                "PARTIALLY_FILLED" if row.filled_quantity else "OPEN"
            )
            refreshed = self._required_row(idempotency_key)
            self.repository.apply_paper_order_update(
                idempotency_key, destination, expected_version=refreshed.version, metadata=metadata,
                detail={"modified": True},
            )
            return self.get_order(idempotency_key)

    def expire_orders(self) -> tuple[PaperOrder, ...]:
        """Expire known timed-out orders when explicitly polled."""

        # Autonomous enumeration is intentionally deferred; quote processing enforces timeout.
        return ()

    def get_order(self, idempotency_key: str) -> PaperOrder:
        """Rebuild the public PAPER order from authoritative persisted state."""

        row = self._required_row(idempotency_key)
        metadata = dict(row.metadata_json)
        fills = tuple(
            PaperFill(
                fill.id, idempotency_key, fill.filled_at, row.symbol, row.instrument_token,
                OrderSide(row.side), fill.quantity, int(metadata.get("lot_size", 1)), fill.price,
                fill.transaction_cost, metadata=fill.metadata_json,
            )
            for fill in self.repository.fills_for_order(idempotency_key)
        )
        status = self._public_status(row.local_state)
        return PaperOrder(
            order_id=idempotency_key, created_at=row.created_at, updated_at=row.updated_at,
            symbol=row.symbol, instrument_token=row.instrument_token,
            strategy_id=str(metadata.get("strategy_id") or "paper-strategy"),
            leg_id=str(metadata.get("leg_id") or "paper-leg"), side=OrderSide(row.side),
            order_type=OrderType(row.order_type), quantity=row.quantity,
            lot_size=int(metadata.get("lot_size", 1)), status=status,
            requested_price=row.limit_price, trigger_price=row.trigger_price,
            fill_price=metadata.get("average_fill_price"), filled_quantity=row.filled_quantity,
            fills=fills, rejection_reason=metadata.get("rejection_reason"),
            remaining_quantity=int(metadata.get("remaining_quantity", row.quantity - row.filled_quantity)),
            average_fill_price=metadata.get("average_fill_price"),
            entry_turnover=float(metadata.get("entry_turnover", 0.0)),
            transaction_costs=float(metadata.get("transaction_costs", 0.0)),
            cancellation_reason=metadata.get("cancellation_reason"), lifecycle_state=row.local_state,
            modification_history=tuple(metadata.get("modification_history", [])), metadata={"paper": True},
        )

    def as_order_response(self, idempotency_key: str) -> OrderResponse:
        """Return the shared order response contract for existing execution consumers."""

        order = self.get_order(idempotency_key)
        return OrderResponse(
            order_id=order.order_id, status=order.status, timestamp=order.updated_at,
            symbol=order.symbol, side=order.side, quantity=order.quantity,
            filled_quantity=order.filled_quantity, average_price=order.average_fill_price,
            rejection_reason=order.rejection_reason, raw_response={"mode": "PAPER"},
        )

    def _required_row(self, key: str):
        row = self.repository.get_order_intent(key)
        if row is None:
            raise PaperBrokerError(f"unknown PAPER order: {key}")
        return row

    def _request_rejection(self, request: OrderRequest) -> str | None:
        if request.order_type not in {OrderType.LIMIT, OrderType.SL_LIMIT, OrderType.MARKET}:
            return "unsupported order type"
        if request.order_type in {OrderType.LIMIT, OrderType.SL_LIMIT} and not request.limit_price:
            return "limit price is required"
        if request.order_type == OrderType.SL_LIMIT and not request.trigger_price:
            return "trigger price is required"
        if request.order_type == OrderType.MARKET and (
            not request.is_emergency_exit or not self.emergency_market_exit_allowed
        ):
            return "MARKET orders require an authorized emergency exit"
        return None

    def _quote_rejection(self, quote: MarketSnapshot, now: datetime) -> str | None:
        if (now - ensure_ist(quote.timestamp)).total_seconds() > self.max_quote_age_seconds:
            return "stale quote"
        if quote.bid is None or quote.ask is None:
            return "bid and ask are required"
        if quote.bid <= 0 or quote.ask <= 0:
            return "bid and ask must be positive"
        if quote.bid > quote.ask:
            return "crossed quote"
        if quote.available_liquidity <= 0:
            return "available liquidity is required"
        return None

    @staticmethod
    def _executable_price(row, quote: MarketSnapshot) -> float | None:
        if row.order_type == OrderType.MARKET.value:
            return quote.ask if row.side == "BUY" else quote.bid
        if row.side == "BUY" and quote.ask <= float(row.limit_price):
            return quote.ask
        if row.side == "SELL" and quote.bid >= float(row.limit_price):
            return quote.bid
        return None

    def _slipped_price(self, price: float, side: str, emergency: bool) -> float:
        percent = self.emergency_slippage_percent if emergency else self.slippage_percent
        multiplier = 1.0 + percent / 100.0 if side == "BUY" else 1.0 - percent / 100.0
        return price * multiplier

    @staticmethod
    def _public_status(state: str) -> OrderStatus:
        mapping = {
            "FILLED": OrderStatus.FILLED, "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "CANCELLED": OrderStatus.CANCELLED, "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED, "SUBMITTED": OrderStatus.SUBMITTED,
        }
        return mapping.get(state, OrderStatus.OPEN)
