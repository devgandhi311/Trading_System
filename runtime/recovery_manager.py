"""Startup recovery for authoritative PAPER runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from runtime.paper_portfolio_service import PaperPortfolioService
from runtime.position_monitor import PositionMonitor
from runtime.reconciliation import PaperReconciliationService
from runtime.session_scheduler import SessionScheduler
from storage.repositories import OperationalRepository


@dataclass(frozen=True)
class RecoveryResult:
    session_id: str | None
    status: str
    orders: int
    fills: int
    positions: int
    pending_exits: int
    exits_resumed: int
    reconciliation_outcome: str | None
    blockers: tuple[str, ...] = ()


class RecoveryManager:
    """Reconstruct state, reconcile uncertainty, and resume safe durable work."""

    def __init__(self, repository: OperationalRepository, scheduler: SessionScheduler,
                 portfolio_service_factory: Callable[[str], PaperPortfolioService],
                 reconciliation_factory: Callable[[str], PaperReconciliationService],
                 position_monitor_factory: Callable[[str, PaperPortfolioService], PositionMonitor],
                 *, broker_order_provider: Callable[[], Any], broker_fill_provider: Callable[[], Any],
                 broker_position_provider: Callable[[], Any]) -> None:
        self.repository = repository
        self.scheduler = scheduler
        self.portfolio_service_factory = portfolio_service_factory
        self.reconciliation_factory = reconciliation_factory
        self.position_monitor_factory = position_monitor_factory
        self.broker_order_provider = broker_order_provider
        self.broker_fill_provider = broker_fill_provider
        self.broker_position_provider = broker_position_provider

    def recover(self) -> RecoveryResult:
        session = self._session()
        if session is None:
            return RecoveryResult(None, "NO_SESSION", 0, 0, 0, 0, 0, None)
        portfolio = self.portfolio_service_factory(session.id)
        portfolio.initialize()
        orders = self.repository.session_orders(session.id)
        fills = self.repository.session_fills(session.id)
        positions = [p for p in self.repository.open_positions() if p.session_id == session.id]
        pending = self.repository.exit_decisions(session.id, pending_only=True)
        blockers: list[str] = []

        reconciliation = self.reconciliation_factory(session.id).reconcile(
            self.broker_order_provider, self.broker_fill_provider, self.broker_position_provider
        )
        outcome = str(getattr(reconciliation.outcome, "value", reconciliation.outcome))
        if reconciliation.entry_blocked:
            blockers.extend(reconciliation.blocking_reasons or (outcome,))
        if any(order.local_state in {"UNKNOWN", "RECONCILIATION_REQUIRED"} for order in orders):
            blockers.append("unknown order state requires reconciliation resolution")
        if any(exit_row.metadata_json.get("status") == "UNKNOWN" for exit_row in pending):
            blockers.append("unknown exit state requires reconciliation resolution")

        resumed = 0
        if outcome == "CLEAN" and not blockers:
            monitor = self.position_monitor_factory(session.id, portfolio)
            resumed_result = monitor.resume_reserved_exits()
            resumed = resumed_result.exits_completed
            blockers.extend(resumed_result.blocked)

        status = "RECOVERED" if not blockers else "BLOCKED"
        self.repository.update_session_metadata(
            session.id,
            recovery={"status": status, "reconciliation_outcome": outcome,
                      "pending_exits": len(pending), "exits_resumed": resumed,
                      "blockers": blockers},
        )
        return RecoveryResult(session.id, status, len(orders), len(fills), len(positions),
                              len(pending), resumed, outcome, tuple(blockers))

    def _session(self):
        open_sessions = self.repository.get_open_sessions()
        if open_sessions:
            session = max(open_sessions, key=lambda row: row.started_at)
            # Rehydrate scheduler state from the same durable session. The scheduler
            # performs its own unresolved-prior-session checks and never opens a
            # second session for the same date/mode.
            resumed = self.scheduler.open_or_resume_session()
            return resumed if resumed.id == session.id else session
        latest = self.repository.get_latest_session()
        if latest is None or latest.status in {"CLOSED", "HOLIDAY"}:
            return None
        return latest
