"""Persistent, restart-safe monitoring of PAPER positions and exits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import logging
from threading import Event
from typing import Any, Callable, Mapping

from runtime.paper_portfolio_service import PaperPortfolioService
from storage.models import PositionRecord
from storage.repositories import OperationalRepository
from utils.time_utils import ensure_ist, now_ist

LOGGER = logging.getLogger(__name__)
TERMINAL_EXIT_STATUSES = {"FILLED", "COMPLETED"}


@dataclass(frozen=True)
class PositionQuote:
    symbol: str
    price: float
    timestamp: datetime


@dataclass(frozen=True)
class MonitorResult:
    inspected: int
    exits_reserved: int
    exits_completed: int
    blocked: tuple[str, ...] = ()


class PositionMonitor:
    """Evaluate every open PAPER position and durably submit each exit once."""

    def __init__(self, repository: OperationalRepository, portfolio_service: PaperPortfolioService,
                 *, session_id: str, quote_provider: Callable[[str], PositionQuote],
                 exit_submitter: Callable[[PositionRecord, str, str, bool], Any],
                 clock: Callable[[], datetime] = now_ist, max_quote_age_seconds: float = 5.0,
                 poll_interval_seconds: float = 1.0) -> None:
        self.repository = repository
        self.portfolio_service = portfolio_service
        self.session_id = session_id
        self.quote_provider = quote_provider
        self.exit_submitter = exit_submitter
        self.clock = clock
        self.max_quote_age_seconds = max_quote_age_seconds
        self.poll_interval_seconds = max(0.1, poll_interval_seconds)
        self.stop_event = Event()

    def request_shutdown(self) -> None:
        self.stop_event.set()

    def run(self, *, max_cycles: int | None = None) -> int:
        cycles = 0
        while not self.stop_event.is_set():
            self.monitor_once()
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            self.stop_event.wait(self.poll_interval_seconds)
        return cycles

    def monitor_once(self) -> MonitorResult:
        now = ensure_ist(self.clock())
        reserved = completed = 0
        blocked: list[str] = []
        positions = [row for row in self.repository.open_positions() if row.session_id == self.session_id]
        for position in positions:
            pending = self._pending_exit(position.id)
            if pending is not None:
                blocked.append(f"{position.id}: pending exit {pending.id}")
                continue
            try:
                quote = self.quote_provider(position.symbol)
                self._validate_quote(quote, now)
                reason, metadata = self._exit_reason(position, quote, now)
                self._persist_monitor_state(position, quote, metadata)
                if reason is None:
                    continue
                exit_id = self.exit_id(position.id)
                decision, created = self.repository.reserve_exit_decision(
                    exit_id=exit_id, session_id=self.session_id, position_id=position.id,
                    reason=reason, decided_at=now,
                    metadata={"quote_price": quote.price, "quote_timestamp": quote.timestamp.isoformat(),
                              "emergency": reason == "EMERGENCY_EXIT"},
                )
                if not created:
                    blocked.append(f"{position.id}: exit already reserved")
                    continue
                reserved += 1
                try:
                    response = self.exit_submitter(position, decision.id, reason, reason == "EMERGENCY_EXIT")
                except Exception as exc:
                    self.repository.update_exit_decision(decision.id, status="UNKNOWN", error=str(exc))
                    blocked.append(f"{position.id}: exit state unknown")
                    continue
                status = self._status(response)
                self.repository.update_exit_decision(decision.id, status=status,
                                                     response_id=self._response_id(response))
                if status in TERMINAL_EXIT_STATUSES:
                    self._complete_position(position, quote, reason, decision.id)
                    completed += 1
                elif status in {"REJECTED", "CANCELLED", "EXPIRED"}:
                    blocked.append(f"{position.id}: exit {status.lower()}")
            except Exception as exc:
                blocked.append(f"{position.id}: {exc}")
                LOGGER.exception("Position monitoring failed closed", extra={"position_id": position.id})
        return MonitorResult(len(positions), reserved, completed, tuple(blocked))

    def resume_reserved_exits(self) -> MonitorResult:
        """Resume only pre-submission RESERVED exits after a clean reconciliation."""
        reserved_rows = [row for row in self.repository.exit_decisions(self.session_id, pending_only=True)
                         if row.metadata_json.get("status") == "RESERVED"]
        completed = 0
        blocked: list[str] = []
        for decision in reserved_rows:
            position = self.repository.get_position(decision.position_id)
            if position is None or position.status == "CLOSED":
                self.repository.update_exit_decision(decision.id, status="COMPLETED", recovery_noop=True)
                continue
            try:
                quote = self.quote_provider(position.symbol)
                self._validate_quote(quote, ensure_ist(self.clock()))
                emergency = bool(decision.metadata_json.get("emergency"))
                response = self.exit_submitter(position, decision.id, decision.reason, emergency)
                status = self._status(response)
                self.repository.update_exit_decision(decision.id, status=status,
                                                     response_id=self._response_id(response), recovered=True)
                if status in TERMINAL_EXIT_STATUSES:
                    self._complete_position(position, quote, decision.reason, decision.id)
                    completed += 1
                else:
                    blocked.append(f"{position.id}: recovered exit {status.lower()}")
            except Exception as exc:
                self.repository.update_exit_decision(decision.id, status="UNKNOWN", error=str(exc), recovered=True)
                blocked.append(f"{position.id}: recovered exit unknown")
        return MonitorResult(len(reserved_rows), 0, completed, tuple(blocked))

    @staticmethod
    def exit_id(position_id: str) -> str:
        return "exit-" + hashlib.sha256(f"{position_id}:close".encode()).hexdigest()[:32]

    def _pending_exit(self, position_id: str):
        exit_id = self.exit_id(position_id)
        row = self.repository.get_exit_decision(exit_id)
        if row and row.metadata_json.get("status") != "COMPLETED":
            return row
        return None

    def _validate_quote(self, quote: PositionQuote, now: datetime) -> None:
        if quote.price <= 0:
            raise ValueError("invalid quote price")
        if (now - ensure_ist(quote.timestamp)).total_seconds() > self.max_quote_age_seconds:
            raise ValueError("stale quote")

    def _exit_reason(self, position: PositionRecord, quote: PositionQuote,
                     now: datetime) -> tuple[str | None, dict[str, Any]]:
        meta = dict(position.metadata_json)
        price = quote.price
        long_position = position.quantity > 0
        high = max(float(meta.get("high_watermark", position.average_price)), price) if long_position else float(meta.get("high_watermark", position.average_price))
        low = min(float(meta.get("low_watermark", position.average_price)), price) if not long_position else float(meta.get("low_watermark", position.average_price))
        updates = {"high_watermark": high, "low_watermark": low, "last_price": price,
                   "last_quote_at": quote.timestamp.isoformat()}
        if meta.get("emergency_exit"):
            return "EMERGENCY_EXIT", updates
        if meta.get("manual_exit"):
            return "MANUAL_EXIT", updates
        expiry = self._date(meta.get("expiry"))
        if expiry is not None and now.date() >= expiry:
            return "EXPIRY_EXIT", updates
        time_exit = self._datetime(meta.get("time_exit_at"))
        if time_exit is not None and now >= time_exit:
            return "TIME_EXIT", updates
        stop = self._float(meta.get("stop_loss"))
        if stop is not None and ((long_position and price <= stop) or (not long_position and price >= stop)):
            return "STOP_LOSS", updates
        target = self._float(meta.get("target"))
        if target is not None and ((long_position and price >= target) or (not long_position and price <= target)):
            return "TARGET", updates
        trailing = self._float(meta.get("trailing_stop_percent"))
        if trailing is not None and trailing > 0:
            trigger = high * (1.0 - trailing / 100.0) if long_position else low * (1.0 + trailing / 100.0)
            updates["trailing_stop_price"] = trigger
            if (long_position and price <= trigger) or (not long_position and price >= trigger):
                return "TRAILING_STOP", updates
        return None, updates

    def _persist_monitor_state(self, position: PositionRecord, quote: PositionQuote,
                               updates: Mapping[str, Any]) -> None:
        meta = {**position.metadata_json, **updates}
        unrealized = (quote.price - position.average_price) * position.quantity
        self.repository.update_position(position.id, expected_version=position.version,
                                        metadata_json=meta, unrealized_pnl=unrealized)

    def _complete_position(self, position: PositionRecord, quote: PositionQuote,
                           reason: str, exit_id: str) -> None:
        current = self.repository.get_position(position.id)
        if current is None or current.status == "CLOSED":
            return
        pnl = (quote.price - current.average_price) * current.quantity
        self.repository.complete_exit_atomic(exit_id=exit_id, exit_price=quote.price, realized_pnl=pnl,
                                             completed_at=ensure_ist(self.clock()))
        self.portfolio_service.initialize()

    @staticmethod
    def _status(response: Any) -> str:
        value = response.get("status") if isinstance(response, Mapping) else getattr(response, "status", None)
        return str(getattr(value, "value", value) or "UNKNOWN").upper()

    @staticmethod
    def _response_id(response: Any) -> str | None:
        value = response.get("order_id") if isinstance(response, Mapping) else getattr(response, "order_id", None)
        return str(value) if value is not None else None

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _date(value: Any) -> date | None:
        try:
            return date.fromisoformat(str(value)) if value else None
        except ValueError:
            return None

    @staticmethod
    def _datetime(value: Any) -> datetime | None:
        try:
            return ensure_ist(datetime.fromisoformat(str(value))) if value else None
        except ValueError:
            return None
