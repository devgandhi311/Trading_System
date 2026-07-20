"""Transactional repositories for authoritative operational state."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from sqlalchemy import desc, select, update
from sqlalchemy.exc import IntegrityError

from storage.database import OperationalDatabase
from storage.models import (
    DecisionContextRecord,
    DecisionCycle,
    ExitDecisionRecord,
    FillRecord,
    KillSwitchEvent,
    MarketSnapshotRecord,
    OrderEvent,
    OrderIntentRecord,
    PortfolioSnapshot,
    PositionRecord,
    ReconciliationEvent,
    RiskDecisionRecord,
    SessionTransitionRecord,
    TradingSession,
)
from models.risk_models import ClearAuthorization, KillSwitchState
from models.paper_models import ReconciliationOutcome
from utils.time_utils import ensure_ist, now_ist

# ---------------------------------------------------------------------------
# Metadata safety
# ---------------------------------------------------------------------------

FORBIDDEN_METADATA_KEYS: set[str] = {
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "passwd",
    "mpin",
    "pin",
    "otp",
    "totp",
    "totp_secret",
    "session_token",
    "bearer",
    "client_secret",
}


class InvalidOrderTransitionError(RuntimeError):
    """Raised when an order state transition is not permitted."""


class OptimisticLockError(RuntimeError):
    """Raised when an optimistic-lock version conflict occurs."""


class ForbiddenSecretMetadataError(ValueError):
    """Raised when metadata contains forbidden secret-like keys."""


def validate_metadata(metadata: Any) -> None:
    """Raise ForbiddenSecretMetadataError if metadata contains a forbidden key at any nesting depth."""
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if isinstance(key, str) and key.casefold() in FORBIDDEN_METADATA_KEYS:
                raise ForbiddenSecretMetadataError(f"Forbidden metadata key: {key!r}")
            validate_metadata(value)
    elif isinstance(metadata, (list, tuple, set)):
        for item in metadata:
            if isinstance(item, str) and item.casefold() in FORBIDDEN_METADATA_KEYS:
                raise ForbiddenSecretMetadataError(f"Forbidden metadata key: {item!r}")
            validate_metadata(item)


# ---------------------------------------------------------------------------
# Decimal quantize helper
# ---------------------------------------------------------------------------

def _quantize(value: float, decimals: int = 2) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0." + "0" * decimals), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Allowed order-state transitions
# ---------------------------------------------------------------------------

ALLOWED_ORDER_TRANSITIONS: dict[str, set[str]] = {
    "CREATED": {"VALIDATED", "REJECTED"},
    "VALIDATED": {"RESERVED", "REJECTED"},
    "REJECTED": set(),
    "RESERVED": {"SUBMITTED", "CANCELLED", "REJECTED"},
    "SUBMITTED": {"IN_FLIGHT", "FILLED", "OPEN", "UNKNOWN", "REJECTED", "PARTIALLY_FILLED", "CANCELLED"},
    "PENDING": {"IN_FLIGHT", "CANCELLED", "REJECTED"},
    "IN_FLIGHT": {"FILLED", "OPEN", "UNKNOWN", "REJECTED", "PARTIALLY_FILLED", "SUBMITTED"},
    "TRIGGER_PENDING": {"FILLED", "OPEN", "UNKNOWN", "CANCELLED", "PARTIALLY_FILLED", "REJECTED", "MODIFY_PENDING", "CANCEL_PENDING", "EXPIRED"},
    "OPEN": {"FILLED", "MODIFY_PENDING", "EXPIRED", "UNKNOWN", "CANCEL_PENDING", "CANCELLED", "PARTIALLY_FILLED", "REJECTED"},
    "PARTIALLY_FILLED": {"FILLED", "MODIFY_PENDING", "EXPIRED", "UNKNOWN", "CANCEL_PENDING", "CANCELLED", "PARTIALLY_FILLED"},
    "MODIFY_PENDING": {"OPEN", "UNKNOWN", "TRIGGER_PENDING", "PARTIALLY_FILLED", "REJECTED"},
    "CANcel_PENDING": {"CANCELLED", "UNKNOWN", "OPEN", "PARTIALLY_FILLED", "TRIGGER_PENDING"},
    "CANCELLED": set(),
    "UNKNOWN": {"FILLED", "OPEN", "RETRY_ALLOWED", "RECONCILIATION_REQUIRED", "PARTIALLY_FILLED", "SUBMITTED"},
    "RECONCILIATION_REQUIRED": {"FILLED", "OPEN", "RETRY_ALLOWED", "PARTIALLY_FILLED", "SUBMITTED"},
    "RETRY_ALLOWED": {"SUBMITTED", "IN_FLIGHT", "FILLED", "OPEN", "PARTIALLY_FILLED", "CANCELLED"},
    "FILLED": set(),
    "EXPIRED": set(),
}
# Normalise the CANcel_PENDING key
ALLOWED_ORDER_TRANSITIONS["CANCEL_PENDING"] = ALLOWED_ORDER_TRANSITIONS.pop("CANcel_PENDING")


# ---------------------------------------------------------------------------
# Portfolio Accounting Repository
# ---------------------------------------------------------------------------

PORTFOLIO_FIELDS = (
    "initial_capital", "current_equity", "realized_pnl", "unrealized_pnl",
    "peak_equity", "accrued_brokerage", "accrued_statutory_costs", "accrued_slippage",
    "margin_used", "capital_at_risk", "margin_reserved", "pending_order_risk",
    "partial_fill_risk", "open_strategy_count",
)

DAILY_COUNTERS = (
    "daily_realized_pnl", "trade_count", "winning_trades", "losing_trades", "consecutive_losses",
)

RECONCILIATION_TOLERANCE = 0.02  # two paise


class PortfolioAccountingRepository:
    """Manages portfolio state, kill-switch events, and equity reconciliation."""

    def __init__(self, database: OperationalDatabase) -> None:
        self.database = database

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state(self, snapshot: PortfolioSnapshot | None, kill_active: bool, kill_reason: str) -> dict[str, Any]:
        """Build a complete portfolio state dictionary from the latest snapshot."""
        state: dict[str, Any] = {field: 0.0 for field in PORTFOLIO_FIELDS}
        state.update({counter: 0 for counter in DAILY_COUNTERS})
        state["_kill_switch_active"] = kill_active
        state["_kill_switch_reason"] = kill_reason
        state["_reconciliation_issues"] = []
        state["version"] = 1

        if snapshot is not None:
            state["initial_capital"] = snapshot.initial_capital
            state["current_equity"] = snapshot.current_equity
            state["realized_pnl"] = snapshot.realized_pnl
            state["unrealized_pnl"] = snapshot.unrealized_pnl
            state["accrued_brokerage"] = snapshot.metadata_json.get("accrued_brokerage", 0.0)
            state["accrued_statutory_costs"] = snapshot.metadata_json.get("accrued_statutory_costs", 0.0)
            state["accrued_slippage"] = snapshot.metadata_json.get("accrued_slippage", 0.0)
            state["peak_equity"] = snapshot.metadata_json.get("peak_equity", snapshot.current_equity)
            state["drawdown"] = snapshot.metadata_json.get("drawdown", 0.0)
            state["margin_used"] = snapshot.metadata_json.get("margin_used", 0.0)
            state["capital_at_risk"] = snapshot.metadata_json.get("capital_at_risk", 0.0)
            state["margin_reserved"] = snapshot.metadata_json.get("margin_reserved", 0.0)
            state["pending_order_risk"] = snapshot.metadata_json.get("pending_order_risk", 0.0)
            state["partial_fill_risk"] = snapshot.metadata_json.get("partial_fill_risk", 0.0)
            state["open_strategy_count"] = snapshot.metadata_json.get("open_strategy_count", 0)
            for counter in DAILY_COUNTERS:
                state[counter] = snapshot.metadata_json.get(counter, 0)
            state["version"] = snapshot.version
        return state

    def _derive_current_equity(self, state: dict[str, Any]) -> float:
        return float(state["initial_capital"]) + float(state["realized_pnl"]) - float(state["accrued_brokerage"]) - float(state["accrued_statutory_costs"]) - float(state["accrued_slippage"])

    def _compute_drawdown(self, current_equity: float, peak_equity: float) -> float:
        if peak_equity <= 0:
            return 0.0
        return (peak_equity - current_equity) / peak_equity * 100.0

    def _compute_available_capital(self, state: dict[str, Any]) -> float:
        return float(state["current_equity"]) - float(state["margin_used"]) - float(state["margin_reserved"])

    def _reconcile_equity(self, state: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        derived = self._derive_current_equity(state)
        stated = float(state["current_equity"])
        if abs(derived - stated) > RECONCILIATION_TOLERANCE:
            issues.append(f"Equity mismatch: derived={derived:.2f} stated={stated:.2f}")
        available = self._compute_available_capital(state)
        if available < 0:
            issues.append(f"Negative available capital: {available:.2f}")
        return issues

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_portfolio_state(self) -> dict[str, Any]:
        """Load the latest portfolio state with kill-switch status."""
        kill_switch = self.current_kill_switch()
        kill_active = bool(kill_switch.active) if kill_switch is not None else False
        kill_reason = str(kill_switch.reason) if kill_switch is not None and kill_switch.reason else ""
        with self.database.transaction() as session:
            snapshot = session.scalar(
                select(PortfolioSnapshot).order_by(desc(PortfolioSnapshot.captured_at)).limit(1)
            )
        state = self._build_state(snapshot, kill_active, kill_reason)
        issues = self._reconcile_equity(state)
        state["_reconciliation_issues"] = issues
        return state

    def save_portfolio_state(self, session_id: str, state: dict[str, Any], *, expected_version: int | None = None) -> dict[str, Any]:
        """Persist a portfolio snapshot with optimistic locking."""
        validate_metadata(state)
        version = expected_version if expected_version is not None else state.get("version", 1)
        derived_equity = self._derive_current_equity(state)
        state["current_equity"] = derived_equity

        existing_peak = float(state.get("peak_equity", derived_equity))
        peak = max(existing_peak, derived_equity)
        drawdown = self._compute_drawdown(derived_equity, peak)
        metadata = {
            "accrued_brokerage": float(state.get("accrued_brokerage", 0.0)),
            "accrued_statutory_costs": float(state.get("accrued_statutory_costs", 0.0)),
            "accrued_slippage": float(state.get("accrued_slippage", 0.0)),
            "peak_equity": peak,
            "drawdown": drawdown,
            "margin_used": float(state.get("margin_used", 0.0)),
            "capital_at_risk": float(state.get("capital_at_risk", 0.0)),
            "margin_reserved": float(state.get("margin_reserved", 0.0)),
            "pending_order_risk": float(state.get("pending_order_risk", 0.0)),
            "partial_fill_risk": float(state.get("partial_fill_risk", 0.0)),
            "open_strategy_count": int(state.get("open_strategy_count", 0)),
        }
        for counter in DAILY_COUNTERS:
            metadata[counter] = state.get(counter, 0)

        with self.database.transaction() as session:
            existing = session.scalar(
                select(PortfolioSnapshot).order_by(desc(PortfolioSnapshot.captured_at)).limit(1)
            )
            if existing is not None:
                if existing.version != version:
                    raise OptimisticLockError(
                        f"Portfolio snapshot version conflict: expected {version}, got {existing.version}"
                    )
            snapshot = PortfolioSnapshot(
                session_id=session_id,
                initial_capital=_quantize(float(state["initial_capital"])),
                current_equity=_quantize(derived_equity),
                realized_pnl=_quantize(float(state.get("realized_pnl", 0.0))),
                unrealized_pnl=_quantize(float(state.get("unrealized_pnl", 0.0))),
                accrued_costs=_quantize(
                    float(state.get("accrued_brokerage", 0.0))
                    + float(state.get("accrued_statutory_costs", 0.0))
                    + float(state.get("accrued_slippage", 0.0))
                ),
                captured_at=now_ist(),
                metadata_json=metadata,
                version=version + 1 if existing is not None else 1,
            )
            session.add(snapshot)
        state["version"] = snapshot.version
        state["peak_equity"] = peak
        state["drawdown"] = drawdown
        issues = self._reconcile_equity(state)
        state["_reconciliation_issues"] = issues
        return state

    def reset_daily_counters(self, session_id: str) -> None:
        """Zero out daily counters in the latest snapshot metadata."""
        with self.database.transaction() as session:
            snapshot = session.scalar(
                select(PortfolioSnapshot).order_by(desc(PortfolioSnapshot.captured_at)).limit(1)
            )
            if snapshot is not None:
                meta = dict(snapshot.metadata_json)
                for counter in DAILY_COUNTERS:
                    meta[counter] = 0
                snapshot.metadata_json = meta

    # ------------------------------------------------------------------
    # Kill-switch
    # ------------------------------------------------------------------

    def current_kill_switch(self) -> KillSwitchState | None:
        """Return the current kill-switch state."""
        with self.database.transaction() as session:
            event = session.scalar(
                select(KillSwitchEvent).order_by(desc(KillSwitchEvent.occurred_at)).limit(1)
            )
        if event is None:
            return None
        return KillSwitchState(
            active=event.active,
            reason=event.reason,
            triggered_at=event.occurred_at if event.active else None,
            source=event.source,
            cleared_at=event.occurred_at if not event.active else None,
            session_id=event.session_id,
            metadata=dict(event.metadata_json),
        )

    def get_active_kill_switch(self) -> KillSwitchState | None:
        """Return the active kill-switch state, if any."""
        with self.database.transaction() as session:
            event = session.scalar(
                select(KillSwitchEvent).where(KillSwitchEvent.active.is_(True)).order_by(desc(KillSwitchEvent.occurred_at)).limit(1)
            )
        if event is None:
            return None
        return KillSwitchState(
            active=event.active,
            reason=event.reason,
            triggered_at=event.occurred_at,
            source=event.source,
            session_id=event.session_id,
            metadata=dict(event.metadata_json),
        )

    def is_kill_switch_active(self) -> bool:
        state = self.get_active_kill_switch()
        if state is None:
            return False
        return state.active

    def trigger_kill_switch(
        self,
        *,
        reason: str,
        source: str,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        triggered_at: datetime | None = None,
    ) -> KillSwitchState:
        """Activate the kill switch and persist an event.

        Repeated triggers are idempotent: if an active kill switch already
        exists, the existing state is returned.
        """
        validate_metadata(metadata)
        existing = self.get_active_kill_switch()
        if existing is not None:
            return existing
        now = triggered_at or now_ist()
        safe_metadata = dict(metadata) if metadata is not None else {}
        with self.database.transaction() as session:
            event = KillSwitchEvent(
                session_id=session_id,
                active=True,
                reason=reason,
                source=source,
                occurred_at=now,
                metadata_json=safe_metadata,
            )
            session.add(event)
        return KillSwitchState(
            active=True,
            reason=reason,
            triggered_at=now,
            source=source,
            session_id=session_id,
            metadata=safe_metadata,
        )

    def clear_kill_switch(
        self,
        *,
        authorization: ClearAuthorization,
        cleared_at: datetime | None = None,
        reason: str | None = None,
        session_id: str | None = None,
    ) -> KillSwitchState:
        """Authorized clear of the kill switch."""
        if not authorization.operator or not authorization.operator.strip():
            raise ValueError("clear_kill_switch requires non-empty authorization operator")
        now = cleared_at or now_ist()
        safe_reason = reason or authorization.reason or "Authorized operator reset"
        auth_meta = {
            "operator": authorization.operator,
            "reason": authorization.reason,
            "authorized_at": authorization.authorized_at.isoformat() if authorization.authorized_at else now.isoformat(),
        }
        with self.database.transaction() as session:
            event = KillSwitchEvent(
                session_id=session_id,
                active=False,
                reason=safe_reason,
                source="authorized_operator",
                occurred_at=now,
                metadata_json=auth_meta,
            )
            session.add(event)
        return KillSwitchState(
            active=False,
            reason=safe_reason,
            triggered_at=None,
            source="authorized_operator",
            cleared_at=now,
            session_id=session_id,
            metadata=auth_meta,
        )

    def kill_switch_history(self) -> list[KillSwitchEvent]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(KillSwitchEvent).order_by(desc(KillSwitchEvent.occurred_at)).limit(100)
            ).all()
        return list(rows)

    # ------------------------------------------------------------------
    # Positions and orders helpers
    # ------------------------------------------------------------------

    def open_positions_with_unrealized_pnl(self) -> list[PositionRecord]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(PositionRecord).where(PositionRecord.status.in_(["OPEN", "PARTIALLY_CLOSED"]))
            ).all()
        return list(rows)

    def pending_orders_with_risk(self) -> list[OrderIntentRecord]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(OrderIntentRecord).where(
                    OrderIntentRecord.local_state.in_(["PENDING", "SUBMITTED", "IN_FLIGHT", "PARTIALLY_FILLED"])
                )
            ).all()
        return list(rows)


# ---------------------------------------------------------------------------
# Operational Repository
# ---------------------------------------------------------------------------


class OperationalRepository:
    """High-level operations over all operational models."""

    def __init__(self, database: OperationalDatabase) -> None:
        self.database = database
        self._portfolio_accounting: PortfolioAccountingRepository | None = None

    # ------------------------------------------------------------------
    # Trading sessions
    # ------------------------------------------------------------------

    def create_trading_session(
        self,
        session_id: str,
        mode: str = "PAPER",
        trading_date: str | None = None,
        *,
        started_at: datetime | None = None,
        status: str = "OPEN",
        metadata: Mapping[str, Any] | None = None,
    ) -> TradingSession:
        """Create or return an existing trading session for the given date and mode.

        One authoritative session exists per trading_date and mode.
        If a duplicate is requested, the existing session is returned.
        """
        from datetime import date
        td = trading_date or date.today().isoformat()
        now = started_at or now_ist()
        safe_metadata = dict(metadata) if metadata is not None else {}
        validate_metadata(safe_metadata)

        existing = self.get_session_by_date_and_mode(td, mode)
        if existing is not None:
            return existing

        record = TradingSession(
            id=session_id,
            mode=mode,
            trading_date=td,
            status=status,
            started_at=now,
            metadata_json=safe_metadata,
        )
        try:
            with self.database.transaction() as session:
                session.add(record)
        except IntegrityError:
            with self.database.transaction() as session:
                existing = session.scalar(
                    select(TradingSession).where(
                        TradingSession.trading_date == td,
                        TradingSession.mode == mode,
                    ).order_by(desc(TradingSession.started_at)).limit(1)
                )
                if existing is not None:
                    return existing
                raise
        return record

    def get_trading_session(self, session_id: str) -> TradingSession | None:
        with self.database.transaction() as session:
            return session.get(TradingSession, session_id)

    def get_session_by_date_and_mode(self, trading_date: str, mode: str) -> TradingSession | None:
        with self.database.transaction() as session:
            return session.scalar(
                select(TradingSession).where(
                    TradingSession.trading_date == trading_date,
                    TradingSession.mode == mode,
                ).order_by(desc(TradingSession.started_at)).limit(1)
            )

    def get_latest_session(self) -> TradingSession | None:
        with self.database.transaction() as session:
            return session.scalar(
                select(TradingSession).order_by(desc(TradingSession.started_at)).limit(1)
            )

    def get_open_sessions(self) -> list[TradingSession]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(TradingSession).where(
                    TradingSession.status.notin_(["CLOSED", "FAILED", "HOLIDAY"])
                )
            ).all()
        return list(rows)

    def update_session_status(
        self,
        session_id: str,
        status: str,
        previous_state: str | None = None,
        expected_version: int | None = None,
        **extra: Any,
    ) -> TradingSession:
        """Update session status with optional optimistic locking."""
        with self.database.transaction() as session:
            record = session.get(TradingSession, session_id)
            if record is None:
                raise ValueError(f"Session {session_id} not found")
            if expected_version is not None and record.version != expected_version:
                raise OptimisticLockError(
                    f"Session {session_id} version conflict: expected {expected_version}, got {record.version}"
                )
            record.status = status
            if previous_state is not None:
                extra["previous_state"] = previous_state
            extra["state_changed_at"] = now_ist().isoformat()
            # Store extras in metadata
            meta = dict(record.metadata_json)
            meta.update(extra)
            validate_metadata(meta)
            record.metadata_json = meta
            record.version += 1
            # Record transition
            detail = {k: v for k, v in extra.items() if k != "reason"}
            validate_metadata(detail)
            transition = SessionTransitionRecord(
                session_id=session_id,
                from_state=previous_state,
                to_state=status,
                reason=extra.get("reason"),
                occurred_at=now_ist(),
                detail=detail,
            )
            session.add(transition)
        return record

    def update_session_metadata(self, session_id: str, **updates: Any) -> TradingSession | None:
        with self.database.transaction() as session:
            record = session.get(TradingSession, session_id)
            if record is None:
                return None
            meta = dict(record.metadata_json)
            meta.update(updates)
            record.metadata_json = meta
            record.version += 1
        return record

    def get_session_transitions(self, session_id: str) -> list[SessionTransitionRecord]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(SessionTransitionRecord)
                .where(SessionTransitionRecord.session_id == session_id)
                .order_by(SessionTransitionRecord.occurred_at)
            ).all()
        return list(rows)

    # ------------------------------------------------------------------
    # Order intents
    # ------------------------------------------------------------------

    def create_order_intent(self, **fields: Any) -> OrderIntentRecord:
        metadata = fields.pop("metadata", None)
        if metadata is not None:
            fields.setdefault("metadata_json", metadata)
        validate_metadata(fields.get("metadata_json", {}))
        with self.database.transaction() as session:
            record = OrderIntentRecord(**fields)
            session.add(record)
        return record

    def reserve_order_intent(self, idempotency_key: str, **fields: Any) -> OrderIntentRecord:
        """Idempotent order-intent reservation using the idempotency key."""
        metadata = fields.pop("metadata", None)
        if metadata is not None:
            fields.setdefault("metadata_json", metadata)
        validate_metadata(fields.get("metadata_json", {}))
        try:
            with self.database.transaction() as session:
                existing = session.scalar(
                    select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key)
                )
                if existing is not None:
                    return existing
                fields.setdefault("idempotency_key", idempotency_key)
                fields.setdefault("client_order_id", f"client-{idempotency_key}")
                fields.setdefault("local_state", "PENDING")
                fields.setdefault("created_at", now_ist())
                fields.setdefault("updated_at", now_ist())
                fields.setdefault("version", 1)
                record = OrderIntentRecord(**fields)
                session.add(record)
                session.flush()
                # Record CREATED event
                session.add(OrderEvent(
                    order_intent_id=record.id,
                    from_state=None,
                    to_state="PENDING",
                    occurred_at=now_ist(),
                    detail={},
                ))
            return record
        except IntegrityError:
            with self.database.transaction() as session:
                existing = session.scalar(
                    select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key)
                )
                if existing is not None:
                    return existing
                raise

    def get_order_intent(self, idempotency_key: str) -> OrderIntentRecord | None:
        with self.database.transaction() as session:
            return session.scalar(
                select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key)
            )

    def transition_order(
        self,
        idempotency_key: str,
        to_state: str,
        expected_version: int,
        **updates: Any,
    ) -> OrderIntentRecord:
        """Transition an order to a new state with optimistic locking."""
        with self.database.transaction() as session:
            record = session.scalar(
                select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key)
            )
            if record is None:
                raise ValueError(f"Order intent {idempotency_key} not found")
            if record.version != expected_version:
                raise OptimisticLockError(
                    f"Order {idempotency_key} version conflict: expected {expected_version}, got {record.version}"
                )
            from_state = record.local_state
            allowed = ALLOWED_ORDER_TRANSITIONS.get(from_state, set())
            if to_state not in allowed and from_state != to_state:
                raise InvalidOrderTransitionError(
                    f"Invalid transition: {from_state} -> {to_state}"
                )
            record.local_state = to_state
            record.version += 1
            record.updated_at = now_ist()
            if "filled_quantity" in updates:
                record.filled_quantity = updates["filled_quantity"]
            if "failure_classification" in updates:
                record.failure_classification = updates["failure_classification"]
            if to_state in ("UNKNOWN", "RECONCILIATION_REQUIRED"):
                record.reconciliation_required = True
            # Record event
            detail = {k: v for k, v in updates.items() if k not in ("expected_version", "filled_quantity", "failure_classification")}
            validate_metadata(detail)
            session.add(OrderEvent(
                order_intent_id=record.id,
                from_state=from_state,
                to_state=to_state,
                occurred_at=now_ist(),
                detail=detail,
            ))
        return record

    def order_events(self, idempotency_key: str) -> list[OrderEvent]:
        with self.database.transaction() as session:
            record = session.scalar(
                select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key)
            )
            if record is None:
                return []
            rows = session.scalars(
                select(OrderEvent).where(OrderEvent.order_intent_id == record.id)
                .order_by(OrderEvent.occurred_at)
            ).all()
        return list(rows)

    # ------------------------------------------------------------------
    # Paper order updates (used by paper_broker)
    # ------------------------------------------------------------------

    def apply_paper_order_update(self, idempotency_key: str, local_state: str | None = None, *, expected_version: int | None = None, **updates: Any) -> OrderIntentRecord | None:
        """Apply an update to a paper order, bypassing transition validation.

        Creates an OrderEvent when local_state changes.
        """
        with self.database.transaction() as session:
            record = session.scalar(
                select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key)
            )
            if record is None:
                return None
            if expected_version is not None and record.version != expected_version:
                raise OptimisticLockError(
                    f"Order {idempotency_key} version conflict: expected {expected_version}, got {record.version}"
                )
            from_state = record.local_state
            if local_state is not None:
                updates["local_state"] = local_state
            metadata = updates.pop("metadata", None)
            if metadata is not None:
                record.metadata_json = dict(metadata)
            detail = updates.pop("detail", None)
            occurred_at = updates.pop("occurred_at", None)
            for key, value in updates.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            record.updated_at = now_ist()
            record.version += 1
            if local_state is not None and from_state != local_state:
                session.add(OrderEvent(
                    order_intent_id=record.id,
                    from_state=from_state,
                    to_state=local_state,
                    occurred_at=occurred_at or now_ist(),
                    detail=dict(detail) if detail is not None else {},
                ))
        return record

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def add_fill(self, fill_id: str, idempotency_key: str, quantity: int, price: float, *, transaction_cost: float = 0.0, filled_at: datetime | None = None, metadata_json: dict[str, Any] | None = None) -> FillRecord:
        with self.database.transaction() as session:
            order = session.scalar(
                select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key)
            )
            if order is None:
                raise ValueError(f"Order {idempotency_key} not found")
            fill = FillRecord(
                id=fill_id,
                order_intent_id=order.id,
                quantity=quantity,
                price=price,
                transaction_cost=transaction_cost,
                filled_at=filled_at or now_ist(),
                metadata_json=metadata_json or {},
            )
            session.add(fill)
        return fill

    def fills_for_order(self, idempotency_key: str) -> list[FillRecord]:
        with self.database.transaction() as session:
            order = session.scalar(
                select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key)
            )
            if order is None:
                return []
            rows = session.scalars(
                select(FillRecord).where(FillRecord.order_intent_id == order.id)
            ).all()
        return list(rows)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def save_position(self, position_id: str, session_id: str, symbol: str, quantity: int, average_price: float,
                      *, strategy_id: str | None = None, metadata: Mapping[str, Any] | None = None) -> PositionRecord:
        safe_metadata = dict(metadata or {})
        validate_metadata(safe_metadata)
        with self.database.transaction() as session:
            existing = session.get(PositionRecord, position_id)
            if existing is not None:
                existing.quantity = quantity
                existing.average_price = average_price
                if strategy_id is not None:
                    existing.strategy_id = strategy_id
                if safe_metadata:
                    existing.metadata_json = {**existing.metadata_json, **safe_metadata}
                existing.updated_at = now_ist()
                existing.version += 1
                return existing
            record = PositionRecord(
                id=position_id,
                session_id=session_id,
                symbol=symbol,
                strategy_id=strategy_id,
                quantity=quantity,
                average_price=average_price,
                status="OPEN",
                updated_at=now_ist(),
                metadata_json=safe_metadata,
            )
            session.add(record)
        return record

    def open_positions(self) -> list[PositionRecord]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(PositionRecord).where(PositionRecord.status.in_(["OPEN", "PARTIALLY_CLOSED"]))
            ).all()
        return list(rows)

    def apply_fill_and_update_order(self, *, fill_id: str, idempotency_key: str, quantity: int,
                                    price: float, transaction_cost: float, new_state: str,
                                    expected_version: int, metadata: Mapping[str, Any],
                                    filled_at: datetime, fail_before_commit: bool = False) -> OrderIntentRecord:
        """Atomically insert a fill and update its order."""
        with self.database.transaction() as session:
            order = session.scalar(select(OrderIntentRecord).where(OrderIntentRecord.idempotency_key == idempotency_key))
            if order is None: raise ValueError(f"Order {idempotency_key} not found")
            if session.get(FillRecord, fill_id) is not None: return order
            if order.version != expected_version: raise OptimisticLockError("fill/order version conflict")
            if quantity <= 0 or order.filled_quantity + quantity > order.quantity: raise ValueError("invalid fill quantity")
            session.add(FillRecord(id=fill_id, order_intent_id=order.id, quantity=quantity, price=price,
                transaction_cost=transaction_cost, filled_at=filled_at, metadata_json={"atomic": True}))
            old = order.local_state; order.filled_quantity += quantity; order.local_state = new_state
            order.metadata_json = dict(metadata); order.updated_at = filled_at; order.version += 1
            session.add(OrderEvent(order_intent_id=order.id, from_state=old, to_state=new_state,
                occurred_at=filled_at, detail={"fill_id": fill_id, "quantity": quantity}))
            if fail_before_commit: raise RuntimeError("injected atomic fill failure")
        return order

    def get_position(self, position_id: str) -> PositionRecord | None:
        """Return one authoritative position."""
        with self.database.transaction() as session:
            return session.get(PositionRecord, position_id)

    def update_position(self, position_id: str, *, expected_version: int | None = None, **updates: Any) -> PositionRecord:
        """Update a position using optimistic locking."""
        with self.database.transaction() as session:
            record = session.get(PositionRecord, position_id)
            if record is None:
                raise ValueError(f"Position {position_id} not found")
            if expected_version is not None and record.version != expected_version:
                raise OptimisticLockError(f"Position {position_id} version conflict")
            metadata = updates.pop("metadata_json", None)
            if metadata is not None:
                validate_metadata(metadata)
                record.metadata_json = dict(metadata)
            for name, value in updates.items():
                if not hasattr(record, name):
                    raise ValueError(f"Unknown position field: {name}")
                setattr(record, name, value)
            record.updated_at = now_ist()
            record.version += 1
        return record

    def reserve_exit_decision(self, *, exit_id: str, session_id: str, position_id: str,
                              reason: str, decided_at: datetime | None = None,
                              metadata: Mapping[str, Any] | None = None) -> tuple[ExitDecisionRecord, bool]:
        """Reserve one exit intent; return the authoritative row and whether it was created."""
        safe_metadata = {"status": "RESERVED", **dict(metadata or {})}
        validate_metadata(safe_metadata)
        existing = self.get_exit_decision(exit_id)
        if existing is not None:
            return existing, False
        record = ExitDecisionRecord(id=exit_id, session_id=session_id, position_id=position_id,
                                    reason=reason, decided_at=decided_at or now_ist(), metadata_json=safe_metadata)
        try:
            with self.database.transaction() as session:
                session.add(record)
        except IntegrityError:
            existing = self.get_exit_decision(exit_id)
            if existing is None:
                raise
            return existing, False
        return record, True

    def get_exit_decision(self, exit_id: str) -> ExitDecisionRecord | None:
        with self.database.transaction() as session:
            return session.get(ExitDecisionRecord, exit_id)

    def exit_decisions(self, session_id: str, *, pending_only: bool = False) -> list[ExitDecisionRecord]:
        with self.database.transaction() as session:
            rows = list(session.scalars(select(ExitDecisionRecord).where(
                ExitDecisionRecord.session_id == session_id).order_by(ExitDecisionRecord.decided_at)).all())
        if pending_only:
            rows = [row for row in rows if row.metadata_json.get("status") not in {"COMPLETED", "CANCELLED"}]
        return rows

    def update_exit_decision(self, exit_id: str, *, status: str, **metadata: Any) -> ExitDecisionRecord:
        validate_metadata(metadata)
        with self.database.transaction() as session:
            record = session.get(ExitDecisionRecord, exit_id)
            if record is None:
                raise ValueError(f"Exit decision {exit_id} not found")
            record.metadata_json = {**record.metadata_json, **metadata, "status": status}
        return record

    def complete_exit_atomic(self, *, exit_id: str, exit_price: float, realized_pnl: float,
                             completed_at: datetime, fail_before_commit: bool = False) -> ExitDecisionRecord:
        """Atomically close a position, update portfolio, and complete its exit."""
        with self.database.transaction() as session:
            decision = session.get(ExitDecisionRecord, exit_id)
            if decision is None: raise ValueError(f"Exit decision {exit_id} not found")
            if decision.metadata_json.get("status") == "COMPLETED": return decision
            position = session.get(PositionRecord, decision.position_id)
            latest = session.scalar(select(PortfolioSnapshot).where(
                PortfolioSnapshot.session_id == decision.session_id
            ).order_by(desc(PortfolioSnapshot.captured_at)).limit(1))
            if position is None or latest is None: raise ValueError("Exit authoritative state unavailable")
            meta = dict(position.metadata_json); risk = float(meta.get("risk_amount", 0)); margin = float(meta.get("margin_required", 0))
            position.status = "CLOSED"; position.quantity = 0; position.realized_pnl = realized_pnl
            position.unrealized_pnl = 0; position.updated_at = completed_at; position.version += 1
            position.metadata_json = {**meta, "exit_id": exit_id, "exit_price": exit_price,
                                      "exit_reason": decision.reason}
            pmeta = dict(latest.metadata_json)
            for key in ("pending_order_risk", "partial_fill_risk", "margin_reserved"): pmeta[key] = 0.0
            pmeta["margin_used"] = max(0.0, float(pmeta.get("margin_used", 0)) - margin)
            pmeta["capital_at_risk"] = max(0.0, float(pmeta.get("capital_at_risk", 0)) - risk)
            pmeta["open_strategy_count"] = max(0, int(pmeta.get("open_strategy_count", 0)) - 1)
            session.add(PortfolioSnapshot(session_id=position.session_id, initial_capital=latest.initial_capital,
                current_equity=latest.current_equity + realized_pnl - latest.unrealized_pnl,
                realized_pnl=latest.realized_pnl + realized_pnl, unrealized_pnl=0,
                accrued_costs=latest.accrued_costs, captured_at=completed_at, metadata_json=pmeta, version=latest.version + 1))
            decision.metadata_json = {**decision.metadata_json, "status": "COMPLETED", "realized_pnl": realized_pnl,
                                      "exit_price": exit_price, "completed_at": completed_at.isoformat()}
            if fail_before_commit: raise RuntimeError("injected atomic exit failure")
        return decision

    def save_market_snapshot(self, *, snapshot_id: str, session_id: str, symbol: str, instrument_token: str,
                             captured_at: datetime, payload: Mapping[str, Any]) -> MarketSnapshotRecord:
        validate_metadata(payload)
        with self.database.transaction() as session:
            row = session.get(MarketSnapshotRecord, snapshot_id)
            if row is None:
                row = MarketSnapshotRecord(id=snapshot_id, session_id=session_id, symbol=symbol,
                    instrument_token=instrument_token, captured_at=captured_at, payload=dict(payload)); session.add(row)
        return row

    def save_risk_decision(self, *, risk_id: str, session_id: str, decision_cycle_id: str | None,
                           approved: bool, reason: str, decided_at: datetime,
                           metadata: Mapping[str, Any]) -> RiskDecisionRecord:
        validate_metadata(metadata)
        with self.database.transaction() as session:
            row = session.get(RiskDecisionRecord, risk_id)
            if row is None:
                row = RiskDecisionRecord(id=risk_id, session_id=session_id, decision_cycle_id=decision_cycle_id,
                    approved=approved, reason=reason, decided_at=decided_at, metadata_json=dict(metadata)); session.add(row)
        return row

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    def save_portfolio_snapshot(
        self,
        session_id: str,
        initial_capital: float,
        current_equity: float,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        accrued_costs: float = 0.0,
    ) -> PortfolioSnapshot:
        with self.database.transaction() as session:
            snapshot = PortfolioSnapshot(
                session_id=session_id,
                initial_capital=_quantize(initial_capital),
                current_equity=_quantize(current_equity),
                realized_pnl=_quantize(realized_pnl),
                unrealized_pnl=_quantize(unrealized_pnl),
                accrued_costs=_quantize(accrued_costs),
                captured_at=now_ist(),
            )
            session.add(snapshot)
        return snapshot

    def latest_portfolio(self) -> PortfolioSnapshot | None:
        with self.database.transaction() as session:
            return session.scalar(
                select(PortfolioSnapshot).order_by(desc(PortfolioSnapshot.captured_at)).limit(1)
            )

    def trading_sessions(self, *, mode: str | None = None) -> list[TradingSession]:
        with self.database.transaction() as session:
            statement = select(TradingSession).order_by(TradingSession.trading_date, TradingSession.started_at)
            if mode is not None:
                statement = statement.where(TradingSession.mode == mode)
            return list(session.scalars(statement).all())

    def latest_portfolio_for_session(self, session_id: str) -> PortfolioSnapshot | None:
        with self.database.transaction() as session:
            return session.scalar(select(PortfolioSnapshot).where(
                PortfolioSnapshot.session_id == session_id
            ).order_by(desc(PortfolioSnapshot.captured_at)).limit(1))

    def portfolio_history(self, session_id: str) -> list[PortfolioSnapshot]:
        with self.database.transaction() as session:
            return list(session.scalars(select(PortfolioSnapshot).where(
                PortfolioSnapshot.session_id == session_id
            ).order_by(PortfolioSnapshot.captured_at)).all())

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def current_kill_switch(self) -> KillSwitchState | None:
        """Return the current kill-switch state."""
        with self.database.transaction() as session:
            event = session.scalar(
                select(KillSwitchEvent).order_by(desc(KillSwitchEvent.occurred_at)).limit(1)
            )
        if event is None:
            return None
        return KillSwitchState(
            active=event.active,
            reason=event.reason,
            triggered_at=event.occurred_at if event.active else None,
            source=event.source,
            cleared_at=event.occurred_at if not event.active else None,
            session_id=event.session_id,
            metadata=dict(event.metadata_json),
        )

    def get_active_kill_switch(self) -> KillSwitchState | None:
        """Return the active kill-switch state, if any."""
        with self.database.transaction() as session:
            event = session.scalar(
                select(KillSwitchEvent).where(KillSwitchEvent.active.is_(True)).order_by(desc(KillSwitchEvent.occurred_at)).limit(1)
            )
        if event is None:
            return None
        return KillSwitchState(
            active=event.active,
            reason=event.reason,
            triggered_at=event.occurred_at,
            source=event.source,
            session_id=event.session_id,
            metadata=dict(event.metadata_json),
        )

    def trigger_kill_switch(
        self,
        *,
        reason: str,
        source: str,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        triggered_at: datetime | None = None,
    ) -> KillSwitchState:
        """Activate the kill switch and persist an event.

        Repeated triggers are idempotent: if an active kill switch already
        exists, the existing state is returned.
        """
        validate_metadata(metadata)
        existing = self.get_active_kill_switch()
        if existing is not None:
            return existing
        now = triggered_at or now_ist()
        safe_metadata = dict(metadata) if metadata is not None else {}
        with self.database.transaction() as session:
            event = KillSwitchEvent(
                session_id=session_id,
                active=True,
                reason=reason,
                source=source,
                occurred_at=now,
                metadata_json=safe_metadata,
            )
            session.add(event)
        return KillSwitchState(
            active=True,
            reason=reason,
            triggered_at=now,
            source=source,
            session_id=session_id,
            metadata=safe_metadata,
        )

    def clear_kill_switch(
        self,
        *,
        authorization: ClearAuthorization,
        cleared_at: datetime | None = None,
        reason: str | None = None,
        session_id: str | None = None,
    ) -> KillSwitchState:
        """Authorized clear of the kill switch."""
        if not authorization.operator or not authorization.operator.strip():
            raise ValueError("clear_kill_switch requires non-empty authorization operator")
        now = cleared_at or now_ist()
        safe_reason = reason or authorization.reason or "Authorized operator reset"
        auth_meta = {
            "operator": authorization.operator,
            "reason": authorization.reason,
            "authorized_at": authorization.authorized_at.isoformat() if authorization.authorized_at else now.isoformat(),
        }
        with self.database.transaction() as session:
            event = KillSwitchEvent(
                session_id=session_id,
                active=False,
                reason=safe_reason,
                source="authorized_operator",
                occurred_at=now,
                metadata_json=auth_meta,
            )
            session.add(event)
        return KillSwitchState(
            active=False,
            reason=safe_reason,
            triggered_at=None,
            source="authorized_operator",
            cleared_at=now,
            session_id=session_id,
            metadata=auth_meta,
        )

    # ------------------------------------------------------------------
    # Session orders and fills
    # ------------------------------------------------------------------

    def session_orders(self, session_id: str) -> list[OrderIntentRecord]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(OrderIntentRecord).where(OrderIntentRecord.session_id == session_id)
            ).all()
        return list(rows)

    def session_fills(self, session_id: str) -> list[FillRecord]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(FillRecord)
                .join(OrderIntentRecord, FillRecord.order_intent_id == OrderIntentRecord.id)
                .where(OrderIntentRecord.session_id == session_id)
            ).all()
        return list(rows)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def save_reconciliation_event(self, result: ReconciliationResult | None = None, **fields: Any) -> ReconciliationEvent:
        """Persist a reconciliation event from a result object or keyword fields."""
        if result is not None:
            fields.setdefault("id", result.reconciliation_id)
            fields.setdefault("session_id", result.session_id)
            fields.setdefault("outcome", result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome))
            fields.setdefault("entry_blocked", result.entry_blocked)
            fields.setdefault("exits_allowed", result.exits_allowed)
            fields.setdefault("occurred_at", result.timestamp)
            fields.setdefault("affected_order_ids", result.affected_order_ids)
            fields.setdefault("affected_position_ids", result.affected_position_ids)
            fields.setdefault("local_state", result.local_state)
            fields.setdefault("broker_state", result.broker_state)
            fields.setdefault("mismatch_details", result.mismatch_details)
            fields.setdefault("reason", result.reason)
            fields.setdefault("valid_until", result.valid_until)
            fields.setdefault("matched_orders", result.matched_orders)
            fields.setdefault("matched_fills", result.matched_fills)
            fields.setdefault("matched_positions", result.matched_positions)
            fields.setdefault("repaired_items", result.repaired_items)
            fields.setdefault("blocking_reasons", result.blocking_reasons)
            fields.setdefault("warnings", result.warnings)
            fields.setdefault("detail", getattr(result, "metadata", {}))
        validate_metadata(fields.get("detail", {}))
        with self.database.transaction() as session:
            record = ReconciliationEvent(**fields)
            session.add(record)
        return record

    def latest_reconciliation(self, session_id: str) -> ReconciliationEvent | None:
        with self.database.transaction() as session:
            return session.scalar(
                select(ReconciliationEvent)
                .where(ReconciliationEvent.session_id == session_id)
                .order_by(desc(ReconciliationEvent.occurred_at)).limit(1)
            )

    def reconciliation_history(self, session_id: str, limit: int = 50) -> list[ReconciliationEvent]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(ReconciliationEvent)
                .where(ReconciliationEvent.session_id == session_id)
                .order_by(desc(ReconciliationEvent.occurred_at)).limit(limit)
            ).all()
        return list(rows)

    def has_unresolved_blocking_reconciliation(self, session_id: str) -> bool:
        latest = self.latest_reconciliation(session_id)
        if latest is None:
            return False
        return latest.entry_blocked

    def is_reconciliation_stale(self, session_id: str, clean_expiry_seconds: int = 300) -> bool:
        latest = self.latest_reconciliation(session_id)
        if latest is None:
            return True
        occurred_at = ensure_ist(latest.occurred_at)
        age = (now_ist() - occurred_at).total_seconds()
        return age > clean_expiry_seconds

    # ------------------------------------------------------------------
    # Decision cycles and decision contexts
    # ------------------------------------------------------------------

    def create_decision_cycle(self, cycle_id: str, session_id: str, *, started_at: datetime | None = None) -> DecisionCycle:
        """Create a new decision cycle record."""
        started_at = started_at or now_ist()
        record = DecisionCycle(
            id=cycle_id,
            session_id=session_id,
            status="STARTED",
            started_at=started_at,
            metadata_json={},
        )
        with self.database.transaction() as session:
            session.add(record)
        return record

    def get_decision_cycle(self, cycle_id: str) -> DecisionCycle | None:
        with self.database.transaction() as session:
            return session.get(DecisionCycle, cycle_id)

    def complete_decision_cycle(self, cycle_id: str, *, completed_at: datetime | None = None) -> DecisionCycle:
        """Mark a decision cycle as completed."""
        completed_at = completed_at or now_ist()
        with self.database.transaction() as session:
            record = session.get(DecisionCycle, cycle_id)
            if record is None:
                raise ValueError(f"Decision cycle {cycle_id} not found")
            record.status = "COMPLETED"
            record.completed_at = completed_at
        return record

    def save_decision_context(self, context: Any) -> DecisionContextRecord:
        """Persist a decision context record."""
        from models.paper_models import DecisionContext
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be a DecisionContext")
        validate_metadata(context.metadata)
        record = DecisionContextRecord(
            id=context.decision_cycle_id,
            decision_cycle_id=context.decision_cycle_id,
            session_id=context.session_id,
            timestamp=context.timestamp,
            market_snapshot_id=context.market_snapshot_id,
            feature_snapshot_id=context.feature_snapshot_id,
            feature_timestamp=context.feature_timestamp,
            regime=context.regime,
            regime_confidence=context.regime_confidence,
            regime_warnings=context.regime_warnings,
            prediction_probabilities=context.prediction_probabilities,
            prediction_confidence=context.prediction_confidence,
            expected_return_percent=context.expected_return_percent,
            expected_range_low=context.expected_range_low,
            expected_range_high=context.expected_range_high,
            model_names=context.model_names,
            model_versions=context.model_versions,
            volatility_forecast=context.volatility_forecast,
            volatility_model_requested=context.volatility_model_requested,
            volatility_model_used=context.volatility_model_used,
            convergence_status=context.convergence_status,
            fallback_used=context.fallback_used,
            fallback_reason=context.fallback_reason,
            expected_expiry_price=context.expected_expiry_price,
            expected_move=context.expected_move,
            range_68=context.range_68,
            range_95=context.range_95,
            tail_risk_score=context.tail_risk_score,
            news_sentiment=context.news_sentiment,
            event_risk_level=context.event_risk_level,
            liquidity_score=context.liquidity_score,
            entry_context_valid=context.entry_context_valid,
            invalid_reason=context.invalid_reason,
            warnings=context.warnings,
            detail=context.metadata,
            version=context.version,
        )
        with self.database.transaction() as session:
            session.add(record)
        return record

    def get_decision_context(self, cycle_id: str) -> DecisionContextRecord | None:
        with self.database.transaction() as session:
            return session.get(DecisionContextRecord, cycle_id)

    def latest_decision_context_for_session(self, session_id: str) -> DecisionContextRecord | None:
        with self.database.transaction() as session:
            return session.scalar(
                select(DecisionContextRecord)
                .where(DecisionContextRecord.session_id == session_id)
                .order_by(desc(DecisionContextRecord.timestamp))
                .limit(1)
            )

    def decision_context_history(self, session_id: str, limit: int = 50) -> list[DecisionContextRecord]:
        with self.database.transaction() as session:
            rows = session.scalars(
                select(DecisionContextRecord)
                .where(DecisionContextRecord.session_id == session_id)
                .order_by(desc(DecisionContextRecord.timestamp))
                .limit(limit)
            ).all()
        return list(rows)

    # ------------------------------------------------------------------
    # Portfolio accounting access
    # ------------------------------------------------------------------

    def get_portfolio_accounting_repo(self) -> PortfolioAccountingRepository:
        if self._portfolio_accounting is None:
            self._portfolio_accounting = PortfolioAccountingRepository(self.database)
        return self._portfolio_accounting
