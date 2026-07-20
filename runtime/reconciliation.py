"""Persistent PAPER reconciliation service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from models.common import OrderStatus
from models.order_models import OrderIntentStatus
from models.paper_models import (
    ReconciliationOutcome,
    ReconciliationResult,
    ENTRY_BLOCKING_OUTCOMES,
    EXIT_ALLOWING_OUTCOMES,
)
from storage.repositories import OperationalRepository
from utils.exceptions import BrokerError
from utils.time_utils import ensure_ist, now_ist

DEFAULT_CLEAN_EXPIRY_SECONDS = 300.0
DEFAULT_EVENTUAL_CONSISTENCY_SECONDS = 120.0

OrderBrokerProvider = Callable[[], Mapping[str, Any]]
FillBrokerProvider = Callable[[], Mapping[str, Any]]
PositionBrokerProvider = Callable[[], Mapping[str, Any]]


class PaperReconciliationService:
    """Compare local PAPER state with broker state and persist results."""

    def __init__(
        self,
        repository: OperationalRepository,
        session_id: str,
        *,
        clean_expiry_seconds: float = DEFAULT_CLEAN_EXPIRY_SECONDS,
        eventual_consistency_seconds: float = DEFAULT_EVENTUAL_CONSISTENCY_SECONDS,
    ) -> None:
        self.repository = repository
        self.session_id = session_id
        self.clean_expiry_seconds = clean_expiry_seconds
        self.eventual_consistency_seconds = eventual_consistency_seconds

    def reconcile(
        self,
        broker_order_provider: OrderBrokerProvider,
        broker_fill_provider: FillBrokerProvider,
        broker_position_provider: PositionBrokerProvider,
        *,
        now: datetime | None = None,
    ) -> ReconciliationResult:
        """Run one reconciliation pass and persist the result."""

        current_time = ensure_ist(now or now_ist())
        local_orders = self._local_orders()
        local_fills = self._local_fills()
        local_positions = self._local_positions()
        broker_orders: dict[str, Any] = {}
        broker_fills: dict[str, Any] = {}
        broker_positions: dict[str, Any] = {}
        broker_available = True
        try:
            broker_orders = broker_order_provider() or {}
        except Exception:
            broker_available = False
        try:
            broker_fills = broker_fill_provider() or {}
        except Exception:
            broker_available = False
        try:
            broker_positions = broker_position_provider() or {}
        except Exception:
            broker_available = False

        affected_order_ids: list[str] = []
        affected_position_ids: list[str] = []
        mismatch_details: list[dict[str, Any]] = []
        blocking_reasons: list[str] = []
        warnings: list[str] = []
        matched_orders = 0
        matched_fills = 0
        matched_positions = 0

        if not broker_available:
            if local_orders or local_fills or local_positions:
                outcome = ReconciliationOutcome.BROKER_UNAVAILABLE
                blocking_reasons.append("Broker providers raised exceptions")
                affected_order_ids = list(local_orders.keys())
                affected_position_ids = list(local_positions.keys())
                mismatch_details.append({"type": "BROKER_UNAVAILABLE", "detail": "all providers failed"})
            else:
                outcome = ReconciliationOutcome.CLEAN
        else:
            outcome, order_matched_orders, order_matched_fills = self._reconcile_orders(
                local_orders=local_orders,
                broker_orders=broker_orders,
                local_fills=local_fills,
                broker_fills=broker_fills,
                affected_order_ids=affected_order_ids,
                mismatch_details=mismatch_details,
                blocking_reasons=blocking_reasons,
                warnings=warnings,
            )
            matched_orders += order_matched_orders
            matched_fills += order_matched_fills
            if outcome is None:
                outcome, position_matched_positions = self._reconcile_positions(
                    local_positions=local_positions,
                    broker_positions=broker_positions,
                    affected_position_ids=affected_position_ids,
                    mismatch_details=mismatch_details,
                    blocking_reasons=blocking_reasons,
                    warnings=warnings,
                )
                matched_positions += position_matched_positions

        if not mismatch_details and outcome is None:
            outcome = ReconciliationOutcome.CLEAN
            matched_orders = len(local_orders)
            matched_fills = sum(len(v) for v in local_fills.values())
            matched_positions = len(local_positions)

        entry_blocked = outcome in ENTRY_BLOCKING_OUTCOMES
        exits_allowed = outcome in EXIT_ALLOWING_OUTCOMES

        valid_until = current_time + timedelta(seconds=self.clean_expiry_seconds)
        if outcome == ReconciliationOutcome.EVENTUAL_CONSISTENCY_PENDING:
            valid_until = current_time + timedelta(seconds=self.eventual_consistency_seconds)
        elif outcome == ReconciliationOutcome.CLEAN:
            valid_until = current_time + timedelta(seconds=self.clean_expiry_seconds)
        elif outcome == ReconciliationOutcome.UNKNOWN_ORDER:
            valid_until = current_time + timedelta(seconds=self.eventual_consistency_seconds)
        else:
            valid_until = None

        reconciliation_id = self._reconciliation_id(current_time)
        result = ReconciliationResult(
            reconciliation_id=reconciliation_id,
            session_id=self.session_id,
            timestamp=current_time,
            outcome=outcome,
            entry_blocked=entry_blocked,
            exits_allowed=exits_allowed,
            affected_order_ids=tuple(affected_order_ids),
            affected_position_ids=tuple(affected_position_ids),
            local_state=self._summarize_local_state(local_orders, local_fills, local_positions),
            broker_state=self._summarize_broker_state(broker_orders, broker_fills, broker_positions),
            mismatch_details=tuple(mismatch_details),
            reason="; ".join(blocking_reasons) if blocking_reasons else "",
            valid_until=valid_until,
            matched_orders=matched_orders,
            matched_fills=matched_fills,
            matched_positions=matched_positions,
            repaired_items=(),
            blocking_reasons=tuple(blocking_reasons),
            warnings=tuple(warnings),
        )
        self.repository.save_reconciliation_event(result)
        return result

    def latest_reconciliation(self) -> ReconciliationResult | None:
        """Return the most recent persisted reconciliation result for this session."""

        row = self.repository.latest_reconciliation(self.session_id)
        if row is None:
            return None
        return self._row_to_result(row)

    def reconciliation_history(self) -> list[ReconciliationResult]:
        """Return all persisted reconciliation results for this session."""

        return [self._row_to_result(row) for row in self.repository.reconciliation_history(self.session_id)]

    def has_unresolved_blocking_reconciliation(self) -> bool:
        """Return true when the latest reconciliation blocks new entries and is unresolved."""

        return self.repository.has_unresolved_blocking_reconciliation(self.session_id)

    def is_reconciliation_stale(self) -> bool:
        """Return true when no recent CLEAN reconciliation exists."""

        return self.repository.is_reconciliation_stale(self.session_id, clean_expiry_seconds=self.clean_expiry_seconds)

    def _reconcile_orders(
        self,
        *,
        local_orders: dict[str, Any],
        broker_orders: dict[str, Any],
        local_fills: dict[str, Any],
        broker_fills: dict[str, Any],
        affected_order_ids: list[str],
        mismatch_details: list[dict[str, Any]],
        blocking_reasons: list[str],
        warnings: list[str],
    ) -> tuple[ReconciliationOutcome | None, int, int]:
        outcome: ReconciliationOutcome | None = None
        eventual_order: str | None = None
        matched_orders = 0
        matched_fills = 0
        mismatch_types: set[str] = set()

        all_keys = set(local_orders) | set(broker_orders)
        for key in all_keys:
            local = local_orders.get(key)
            broker = broker_orders.get(key)
            if local and not broker:
                affected_order_ids.append(key)
                mismatch_details.append({"type": "LOCAL_ORDER_MISSING", "order_id": key, "local_state": self._order_summary(local)})
                blocking_reasons.append(f"Local order {key} missing from broker")
                mismatch_types.add("LOCAL_ORDER_MISSING")
                outcome = ReconciliationOutcome.LOCAL_ORDER_MISSING
                continue
            if broker and not local:
                affected_order_ids.append(key)
                mismatch_details.append({"type": "BROKER_ORDER_MISSING", "order_id": key, "broker_state": self._order_summary(broker)})
                blocking_reasons.append(f"Broker order {key} missing locally")
                mismatch_types.add("BROKER_ORDER_MISSING")
                outcome = ReconciliationOutcome.BROKER_ORDER_MISSING
                continue

            matched_orders += 1
            local_status = str(getattr(local, "local_state", "") or "")
            broker_status = str(getattr(broker, "lifecycle_state", "") or getattr(broker, "status", "") or "")
            local_recon_required = bool(getattr(local, "reconciliation_required", False))
            local_intent_status = getattr(local, "intent_status", None)

            if local_status in {"UNKNOWN", "RECONCILIATION_REQUIRED"} or local_recon_required or (isinstance(local_intent_status, OrderIntentStatus) and local_intent_status in {OrderIntentStatus.UNKNOWN, OrderIntentStatus.RECONCILIATION_REQUIRED}):
                affected_order_ids.append(key)
                mismatch_details.append({"type": "UNKNOWN_ORDER", "order_id": key, "local_status": local_status, "broker_status": broker_status})
                blocking_reasons.append(f"Order {key} is in UNKNOWN or RECONCILIATION_REQUIRED state")
                mismatch_types.add("UNKNOWN_ORDER")
                outcome = ReconciliationOutcome.UNKNOWN_ORDER
                continue

            transitional_states = {"SUBMITTED", "IN_FLIGHT", "TRIGGER_PENDING", "MODIFY_PENDING", "CANCEL_PENDING"}
            if local_status in transitional_states or broker_status in transitional_states:
                eventual_order = key
                continue

            if local_status != broker_status:
                affected_order_ids.append(key)
                mismatch_details.append({"type": "ORDER_STATE_MISMATCH", "order_id": key, "local_state": local_status, "broker_state": broker_status})
                blocking_reasons.append(f"Order {key} state mismatch: {local_status} != {broker_status}")
                mismatch_types.add("ORDER_STATE_MISMATCH")
                outcome = ReconciliationOutcome.ORDER_STATE_MISMATCH

            local_qty = int(getattr(local, "quantity", 0) or 0)
            broker_qty = int(getattr(broker, "quantity", 0) or 0)
            if local_qty != broker_qty:
                affected_order_ids.append(key)
                mismatch_details.append({"type": "QUANTITY_MISMATCH", "order_id": key, "local_quantity": local_qty, "broker_quantity": broker_qty})
                blocking_reasons.append(f"Order {key} quantity mismatch: {local_qty} != {broker_qty}")
                mismatch_types.add("QUANTITY_MISMATCH")
                outcome = ReconciliationOutcome.QUANTITY_MISMATCH

            local_filled = int(getattr(local, "filled_quantity", 0) or 0)
            broker_filled = int(getattr(broker, "filled_quantity", 0) or 0)
            if local_filled != broker_filled:
                affected_order_ids.append(key)
                mismatch_details.append({"type": "FILL_MISMATCH", "order_id": key, "local_filled_quantity": local_filled, "broker_filled_quantity": broker_filled})
                blocking_reasons.append(f"Order {key} filled quantity mismatch: {local_filled} != {broker_filled}")
                mismatch_types.add("FILL_MISMATCH")
                outcome = ReconciliationOutcome.FILL_MISMATCH

            local_avg = getattr(local, "average_fill_price", None)
            if local_avg is None and hasattr(local, "metadata_json"):
                local_avg = local.metadata_json.get("average_fill_price")
            broker_avg = getattr(broker, "average_fill_price", None)
            if local_avg is not None and broker_avg is not None and abs(float(local_avg) - float(broker_avg)) > 1e-6:
                affected_order_ids.append(key)
                mismatch_details.append({"type": "PRICE_MISMATCH", "order_id": key, "local_average_fill_price": local_avg, "broker_average_fill_price": broker_avg})
                blocking_reasons.append(f"Order {key} average fill price mismatch: {local_avg} != {broker_avg}")
                mismatch_types.add("PRICE_MISMATCH")
                outcome = ReconciliationOutcome.PRICE_MISMATCH

            local_costs = getattr(local, "transaction_costs", 0.0)
            if local_costs == 0.0 and hasattr(local, "metadata_json"):
                local_costs = float(local.metadata_json.get("transaction_costs", 0.0))
            broker_costs = float(getattr(broker, "transaction_costs", 0.0) or 0.0)
            if abs(local_costs - broker_costs) > 1e-6:
                affected_order_ids.append(key)
                mismatch_details.append({"type": "COST_MISMATCH", "order_id": key, "local_transaction_costs": local_costs, "broker_transaction_costs": broker_costs})
                blocking_reasons.append(f"Order {key} transaction cost mismatch: {local_costs} != {broker_costs}")
                mismatch_types.add("COST_MISMATCH")
                outcome = ReconciliationOutcome.COST_MISMATCH

            local_fill_list = local_fills.get(key, [])
            broker_fill_list = broker_fills.get(key, [])
            if len(local_fill_list) != len(broker_fill_list):
                affected_order_ids.append(key)
                mismatch_details.append({"type": "FILL_MISMATCH", "order_id": key, "local_fill_count": len(local_fill_list), "broker_fill_count": len(broker_fill_list)})
                blocking_reasons.append(f"Order {key} fill count mismatch: {len(local_fill_list)} != {len(broker_fill_list)}")
                mismatch_types.add("FILL_MISMATCH")
                outcome = ReconciliationOutcome.FILL_MISMATCH
            else:
                for idx, (lfill, bfill) in enumerate(zip(local_fill_list, broker_fill_list)):
                    if abs(float(getattr(lfill, "quantity", 0) or 0) - float(getattr(bfill, "quantity", 0) or 0)) > 1e-6:
                        affected_order_ids.append(key)
                        mismatch_details.append({"type": "FILL_MISMATCH", "order_id": key, "fill_index": idx, "local_quantity": getattr(lfill, "quantity", 0), "broker_quantity": getattr(bfill, "quantity", 0)})
                        blocking_reasons.append(f"Order {key} fill {idx} quantity mismatch")
                        mismatch_types.add("FILL_MISMATCH")
                        outcome = ReconciliationOutcome.FILL_MISMATCH
                    if abs(float(getattr(lfill, "price", 0) or 0) - float(getattr(bfill, "price", 0) or 0)) > 1e-6:
                        affected_order_ids.append(key)
                        mismatch_details.append({"type": "PRICE_MISMATCH", "order_id": key, "fill_index": idx, "local_price": getattr(lfill, "price", 0), "broker_price": getattr(bfill, "price", 0)})
                        blocking_reasons.append(f"Order {key} fill {idx} price mismatch")
                        mismatch_types.add("PRICE_MISMATCH")
                        outcome = ReconciliationOutcome.PRICE_MISMATCH

            matched_fills += min(len(local_fill_list), len(broker_fill_list))

        if len(mismatch_types) > 1:
            outcome = ReconciliationOutcome.AMBIGUOUS
        elif outcome is None and eventual_order is not None:
            outcome = ReconciliationOutcome.EVENTUAL_CONSISTENCY_PENDING
            warnings.append(f"Order {eventual_order} is in a transitional state")
        elif outcome is None and not all_keys:
            outcome = None

        return outcome, matched_orders, matched_fills

    def _reconcile_positions(
        self,
        *,
        local_positions: dict[str, Any],
        broker_positions: dict[str, Any],
        affected_position_ids: list[str],
        mismatch_details: list[dict[str, Any]],
        blocking_reasons: list[str],
        warnings: list[str],
    ) -> tuple[ReconciliationOutcome | None, int]:
        outcome: ReconciliationOutcome | None = None
        matched_positions = 0
        mismatch_types: set[str] = set()
        all_keys = set(local_positions) | set(broker_positions)
        for key in all_keys:
            local = local_positions.get(key)
            broker = broker_positions.get(key)
            if local and not broker:
                affected_position_ids.append(key)
                mismatch_details.append({"type": "LOCAL_POSITION_MISSING", "position_id": key})
                blocking_reasons.append(f"Local position {key} missing from broker")
                mismatch_types.add("LOCAL_POSITION_MISSING")
                outcome = ReconciliationOutcome.POSITION_MISMATCH
                continue
            if broker and not local:
                affected_position_ids.append(key)
                mismatch_details.append({"type": "BROKER_POSITION_MISSING", "position_id": key})
                blocking_reasons.append(f"Broker position {key} missing locally")
                mismatch_types.add("BROKER_POSITION_MISSING")
                outcome = ReconciliationOutcome.POSITION_MISMATCH
                continue

            matched_positions += 1
            local_qty = int(getattr(local, "quantity", 0) or 0)
            broker_qty = int(getattr(broker, "quantity", 0) or 0)
            if local_qty != broker_qty:
                affected_position_ids.append(key)
                mismatch_details.append({"type": "POSITION_MISMATCH", "position_id": key, "local_quantity": local_qty, "broker_quantity": broker_qty})
                blocking_reasons.append(f"Position {key} quantity mismatch: {local_qty} != {broker_qty}")
                mismatch_types.add("POSITION_MISMATCH")
                outcome = ReconciliationOutcome.POSITION_MISMATCH

            local_avg = float(getattr(local, "average_price", 0.0) or 0.0)
            broker_avg = float(getattr(broker, "entry_price", getattr(broker, "average_price", 0.0)) or 0.0)
            if abs(local_avg - broker_avg) > 1e-6:
                affected_position_ids.append(key)
                mismatch_details.append({"type": "PRICE_MISMATCH", "position_id": key, "local_average_price": local_avg, "broker_average_price": broker_avg})
                blocking_reasons.append(f"Position {key} average price mismatch: {local_avg} != {broker_avg}")
                mismatch_types.add("POSITION_MISMATCH")
                outcome = ReconciliationOutcome.POSITION_MISMATCH

            local_pnl = float(getattr(local, "realized_pnl", 0.0) or 0.0)
            broker_pnl = float(getattr(broker, "realized_pnl", 0.0) or 0.0)
            if abs(local_pnl - broker_pnl) > 1e-6:
                affected_position_ids.append(key)
                mismatch_details.append({"type": "POSITION_MISMATCH", "position_id": key, "local_realized_pnl": local_pnl, "broker_realized_pnl": broker_pnl})
                blocking_reasons.append(f"Position {key} realized P&L mismatch: {local_pnl} != {broker_pnl}")
                mismatch_types.add("POSITION_MISMATCH")
                outcome = ReconciliationOutcome.POSITION_MISMATCH

        if len(mismatch_types) > 1:
            outcome = ReconciliationOutcome.AMBIGUOUS
        elif outcome is None and not all_keys:
            outcome = None

        return outcome, matched_positions

    def _local_orders(self) -> dict[str, Any]:
        rows = self.repository.session_orders(self.session_id)
        return {row.idempotency_key: row for row in rows}

    def _local_fills(self) -> dict[str, list[Any]]:
        rows = self.repository.session_fills(self.session_id)
        fills: dict[str, list[Any]] = {}
        for fill in rows:
            intent_id = fill.order_intent_id
            if intent_id not in fills:
                fills[intent_id] = []
            fills[intent_id].append(fill)
        return fills

    def _local_positions(self) -> dict[str, Any]:
        rows = [row for row in self.repository.open_positions() if row.session_id == self.session_id]
        return {row.id: row for row in rows}

    def _summarize_local_state(
        self,
        orders: dict[str, Any],
        fills: dict[str, Any],
        positions: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "order_count": len(orders),
            "fill_count": sum(len(v) for v in fills.values()),
            "position_count": len(positions),
            "orders": {k: self._order_summary(v) for k, v in orders.items()},
        }

    def _summarize_broker_state(
        self,
        orders: dict[str, Any],
        fills: dict[str, Any],
        positions: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "order_count": len(orders),
            "fill_count": sum(len(v) for v in fills.values()),
            "position_count": len(positions),
            "orders": {k: self._order_summary(v) for k, v in orders.items()},
        }

    @staticmethod
    def _order_summary(order: Any) -> dict[str, Any]:
        return {
            "status": str(getattr(order, "local_state", "") or getattr(order, "lifecycle_state", "") or getattr(order, "status", "")),
            "quantity": int(getattr(order, "quantity", 0) or 0),
            "filled_quantity": int(getattr(order, "filled_quantity", 0) or 0),
            "average_fill_price": getattr(order, "average_fill_price", None),
            "transaction_costs": float(getattr(order, "transaction_costs", 0.0) or 0.0),
        }

    @staticmethod
    def _reconciliation_id(timestamp: datetime) -> str:
        return f"recon-{timestamp.strftime('%Y%m%d%H%M%S%f')}"

    @staticmethod
    def _row_to_result(row: Any) -> ReconciliationResult:
        detail = row.detail or {}
        return ReconciliationResult(
            reconciliation_id=row.id,
            session_id=row.session_id,
            timestamp=row.occurred_at,
            outcome=ReconciliationOutcome(row.outcome),
            entry_blocked=row.entry_blocked,
            exits_allowed=row.exits_allowed,
            affected_order_ids=tuple(row.affected_order_ids or ()),
            affected_position_ids=tuple(row.affected_position_ids or ()),
            local_state=dict(row.local_state or {}),
            broker_state=dict(row.broker_state or {}),
            mismatch_details=tuple(row.mismatch_details or ()),
            reason=str(row.reason or ""),
            valid_until=row.valid_until,
            matched_orders=row.matched_orders,
            matched_fills=row.matched_fills,
            matched_positions=row.matched_positions,
            repaired_items=tuple(row.repaired_items or ()),
            blocking_reasons=tuple(row.blocking_reasons or ()),
            warnings=tuple(row.warnings or ()),
        )
