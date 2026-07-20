"""Durable PAPER session scheduler and lifecycle controller.

This module implements the NSE session lifecycle with strict state management,
persistent transitions, clock validation, and concurrency-safe session creation.

All times are Asia/Kolkata. An injected clock (callable that returns datetime)
allows deterministic testing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from models.paper_models import SessionState
from storage.database import OperationalDatabase
from storage.models import (
    KillSwitchEvent,
    OrderIntentRecord,
    PositionRecord,
    ReconciliationEvent,
    SessionTransitionRecord,
    TradingSession,
)
from storage.repositories import OperationalRepository, OptimisticLockError
from utils.time_utils import combine_ist, ensure_ist, now_ist, parse_hhmm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")

ALLOWED_STATE_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.NOT_STARTED: {SessionState.PREOPEN, SessionState.HOLIDAY, SessionState.RECOVERY_REQUIRED, SessionState.FAILED},
    SessionState.PREOPEN: {SessionState.OPEN, SessionState.ENTRY_ALLOWED, SessionState.HOLIDAY, SessionState.FAILED},
    SessionState.OPEN: {SessionState.ENTRY_ALLOWED, SessionState.EXIT_ONLY, SessionState.SQUAREOFF, SessionState.CLOSED, SessionState.FAILED},
    SessionState.ENTRY_ALLOWED: {SessionState.EXIT_ONLY, SessionState.SQUAREOFF, SessionState.CLOSED, SessionState.FAILED},
    SessionState.EXIT_ONLY: {SessionState.SQUAREOFF, SessionState.CLOSED, SessionState.FAILED},
    SessionState.SQUAREOFF: {SessionState.CLOSED, SessionState.FAILED},
    SessionState.CLOSED: set(),
    SessionState.HOLIDAY: {SessionState.NOT_STARTED, SessionState.CLOSED},
    SessionState.FAILED: {SessionState.RECOVERY_REQUIRED},
    SessionState.RECOVERY_REQUIRED: {SessionState.PREOPEN, SessionState.FAILED, SessionState.CLOSED},
}

# Maximum allowed clock drift in seconds when a reference time is provided
MAX_CLOCK_DRIFT_SECONDS = 60.0

# Minimum gap between state polls in seconds
MIN_POLL_INTERVAL_SECONDS = 1.0


@dataclass
class SchedulerState:
    """In-memory scheduler runtime state."""
    session_id: str | None = None
    current_session: TradingSession | None = None
    current_state: SessionState = SessionState.NOT_STARTED
    previous_state: SessionState | None = None
    state_changed_at: datetime | None = None
    last_poll_time: datetime | None = None
    _initialized: bool = False


# ---------------------------------------------------------------------------
# Session Scheduler
# ---------------------------------------------------------------------------


class SessionScheduler:
    """Manage the NSE session lifecycle with persistence and clock validation.

    Parameters
    ----------
    database : OperationalDatabase
        The authoritative operational database.
    mode : str
        Trading mode (e.g. "PAPER", "LIVE").
    clock : Callable[[], datetime] | None
        Injected clock returning Asia/Kolkata-aware datetime. Defaults to
        ``utils.time_utils.now_ist``.
    market_start : str | time
        HH:MM market open time. Default 09:15.
    market_end : str | time
        HH:MM market close time. Default 15:30.
    no_new_entry_cutoff : str | time
        HH:MM after which new entries are blocked. Default 15:00.
    squareoff_start : str | time
        HH:MM at which square-off starts. Default 15:20.
    holidays : set[date] | None
        Set of configured NSE holidays. Defaults to empty set.
    max_drift_seconds : float
        Maximum allowed clock drift from reference time. Default 60.
    """

    def __init__(
        self,
        database: OperationalDatabase,
        mode: str = "PAPER",
        *,
        clock: Callable[[], datetime] | None = None,
        market_start: str | time = time(9, 15),
        market_end: str | time = time(15, 30),
        no_new_entry_cutoff: str | time = time(15, 0),
        squareoff_start: str | time = time(15, 20),
        holidays: set[date] | None = None,
        max_drift_seconds: float = MAX_CLOCK_DRIFT_SECONDS,
    ) -> None:
        self.database = database
        self.mode = mode
        self._clock = clock or now_ist
        self.market_start = parse_hhmm(market_start) if isinstance(market_start, str) else market_start
        self.market_end = parse_hhmm(market_end) if isinstance(market_end, str) else market_end
        self.no_new_entry_cutoff = parse_hhmm(no_new_entry_cutoff) if isinstance(no_new_entry_cutoff, str) else no_new_entry_cutoff
        self.squareoff_start = parse_hhmm(squareoff_start) if isinstance(squareoff_start, str) else squareoff_start
        self.holidays = holidays or set()
        self.max_drift_seconds = max_drift_seconds
        self.repo = OperationalRepository(database)
        self._state = SchedulerState()

    # ------------------------------------------------------------------
    # Clock helpers
    # ------------------------------------------------------------------

    def _now(self) -> datetime:
        """Return the current injected-clock time in Asia/Kolkata."""
        return ensure_ist(self._clock())

    def _today(self) -> date:
        return self._now().date()

    def _market_open_time(self, day: date | None = None) -> datetime:
        return combine_ist(day or self._today(), self.market_start)

    def _market_close_time(self, day: date | None = None) -> datetime:
        return combine_ist(day or self._today(), self.market_end)

    def _entry_cutoff_time(self, day: date | None = None) -> datetime:
        return combine_ist(day or self._today(), self.no_new_entry_cutoff)

    def _squareoff_time(self, day: date | None = None) -> datetime:
        return combine_ist(day or self._today(), self.squareoff_start)

    def _is_weekday(self, day: date | None = None) -> bool:
        return (day or self._today()).weekday() < 5

    def _is_holiday(self, day: date | None = None) -> bool:
        return (day or self._today()) in self.holidays

    @staticmethod
    def _session_id(trading_date: date, mode: str) -> str:
        return f"{mode}_{trading_date.isoformat()}"

    # ------------------------------------------------------------------
    # Clock validation
    # ------------------------------------------------------------------

    def _check_backward_clock(self, now: datetime) -> None:
        """Detect backward clock movement since the last poll."""
        if self._state.last_poll_time is not None:
            if now < self._state.last_poll_time:
                raise RuntimeError(
                    f"Backward clock movement detected: "
                    f"previous={self._state.last_poll_time.isoformat()}, "
                    f"now={now.isoformat()}"
                )

    def _check_clock_drift(self, now: datetime, reference_time: datetime | None = None) -> None:
        """Raise if clock drift from reference exceeds max_drift_seconds."""
        if reference_time is not None:
            drift = abs((now - reference_time).total_seconds())
            if drift > self.max_drift_seconds:
                raise RuntimeError(
                    f"Clock drift violation: {drift:.2f}s exceeds "
                    f"max {self.max_drift_seconds}s"
                )

    # ------------------------------------------------------------------
    # Phase determination
    # ------------------------------------------------------------------

    def _determine_phase(self, now: datetime) -> SessionState:
        """Determine the session phase based on the current clock time."""
        today = now.date()

        # Weekend -> HOLIDAY
        if not self._is_weekday(today):
            return SessionState.HOLIDAY

        # Configured holiday -> HOLIDAY
        if self._is_holiday(today):
            return SessionState.HOLIDAY

        current_time = now.time()

        # Before market open -> PREOPEN (allow pre-market checks from midnight)
        if current_time < self.market_start:
            return SessionState.PREOPEN

        # At exact market start -> ENTRY_ALLOWED (market is open)
        # Entry-cutoff inclusive: at 15:00 exactly, entries are still allowed
        if current_time <= self.no_new_entry_cutoff:
            return SessionState.ENTRY_ALLOWED

        # Entry cutoff passed -> EXIT_ONLY
        if current_time < self.squareoff_start:
            return SessionState.EXIT_ONLY

        # Square-off window -> SQUAREOFF
        if current_time < self.market_end:
            return SessionState.SQUAREOFF

        # Market closed -> CLOSED
        return SessionState.CLOSED

    # ------------------------------------------------------------------
    # Session resolution helpers
    # ------------------------------------------------------------------

    def _has_unresolved_previous_session(self, today: date) -> TradingSession | None:
        """Check if a prior session for the same mode has an unresolved state.

        An unresolved session has status in {OPEN, PREOPEN, ENTRY_ALLOWED,
        EXIT_ONLY, SQUAREOFF, FAILED, RECOVERY_REQUIRED} and belongs to an
        earlier trading date.
        """
        for session in self.repo.get_open_sessions():
            if session.mode != self.mode:
                continue
            session_date = session.trading_date
            if session_date < today.isoformat():
                return session
        return None

    def _has_unknown_orders(self, session_id: str) -> bool:
        """Check if the given session has orders in UNKNOWN state."""
        orders = self.repo.session_orders(session_id)
        return any(o.local_state == "UNKNOWN" for o in orders)

    def _has_unresolved_reconciliation(self, session_id: str) -> bool:
        """Check if the session has a blocking reconciliation."""
        return self.repo.has_unresolved_blocking_reconciliation(session_id)

    def _has_open_positions(self, session_id: str) -> bool:
        """Check if there are open positions for this session."""
        for pos in self.repo.open_positions():
            if pos.session_id == session_id:
                return True
        return False

    def _has_open_positions_for_today(self) -> bool:
        """Check if there are open positions not yet resolved."""
        return len(self.repo.open_positions()) > 0

    def _is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        ks = self.repo.current_kill_switch()
        return ks is not None and ks.active

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_or_resume_session(self) -> TradingSession:
        """Open a new session for today or resume an existing active session.

        Returns the authoritative TradingSession record.

        Raises
        ------
        RuntimeError
            If clock drift or backward movement is detected.
        RuntimeError
            If an unresolved prior session blocks opening.
        """
        now = self._now()
        self._check_backward_clock(now)

        today = now.date()
        session_id = self._session_id(today, self.mode)

        # Check if a session already exists for today
        existing = self.repo.get_session_by_date_and_mode(today.isoformat(), self.mode)
        if existing is not None:
            if existing.status in ("CLOSED", "HOLIDAY", "FAILED"):
                # Already ended; reopen as RECOVERY_REQUIRED if not CLOSED cleanly
                if existing.status == "CLOSED":
                    return existing
                return self.mark_recovery_required(
                    existing.id,
                    reason=f"Session {existing.id} ended as {existing.status}"
                )
            # Resume existing session
            self._state.session_id = existing.id
            self._state.current_session = existing
            self._state.current_state = SessionState(existing.status)
            self._state.previous_state = SessionState(existing.previous_state) if existing.previous_state else None
            self._state.state_changed_at = existing.state_changed_at
            self._state._initialized = True
            self._state.last_poll_time = now
            return existing

        # Check for unresolved previous session
        unresolved = self._has_unresolved_previous_session(today)
        if unresolved is not None:
            # Transition the unresolved session to RECOVERY_REQUIRED
            return self.mark_recovery_required(
                unresolved.id,
                reason=f"Prior session {unresolved.id} is unresolved"
            )

        # Determine the phase
        phase = self._determine_phase(now)

        # Create the session
        new_record = TradingSession(
            id=session_id,
            mode=self.mode,
            trading_date=today.isoformat(),
            status=phase.value,
            started_at=now,
            previous_state=None,
            state_changed_at=now,
        )
        try:
            with self.database.transaction() as session:
                # Use insert-or-fail via unique constraint on (trading_date, mode)
                session.add(new_record)
                session.flush()
                # Record initial transition
                transition = SessionTransitionRecord(
                    session_id=session_id,
                    from_state=None,
                    to_state=phase.value,
                    reason="Session opened",
                    occurred_at=now,
                    detail={"mode": self.mode},
                )
                session.add(transition)
        except Exception:
            # Unique constraint violation - another writer created it
            existing = self.repo.get_session_by_date_and_mode(today.isoformat(), self.mode)
            if existing is not None:
                self._state.session_id = existing.id
                self._state.current_session = existing
                self._state.current_state = SessionState(existing.status)
                self._state._initialized = True
                self._state.last_poll_time = now
                return existing
            raise

        self._state.session_id = session_id
        self._state.current_session = new_record
        self._state.current_state = phase
        self._state.previous_state = None
        self._state.state_changed_at = now
        self._state._initialized = True
        self._state.last_poll_time = now
        return new_record

    def get_current_phase(self) -> SessionState:
        """Return the current session phase based on clock time.

        Returns the persisted session state if a session is active,
        otherwise the clock-determined phase.
        """
        if self._state.current_session is not None:
            return self._state.current_state
        return self._determine_phase(self._now())

    def entry_allowed(self) -> bool:
        """Return True if new entries are permitted."""
        if not self._state._initialized:
            return False
        if self._is_kill_switch_active():
            return False
        return self._state.current_state == SessionState.ENTRY_ALLOWED

    def exits_allowed(self) -> bool:
        """Return True if exits are permitted (positions can be reduced/closed)."""
        if not self._state._initialized:
            return False
        return self._state.current_state in {
            SessionState.OPEN,
            SessionState.ENTRY_ALLOWED,
            SessionState.EXIT_ONLY,
            SessionState.SQUAREOFF,
        }

    def squareoff_required(self) -> bool:
        """Return True if square-off is required (positions must be closed)."""
        if not self._state._initialized:
            return False
        return self._state.current_state in {
            SessionState.SQUAREOFF,
            SessionState.EXIT_ONLY,
        }

    def close_session(
        self,
        session_id: str | None = None,
        *,
        final_equity: float | None = None,
        reason: str = "Session closed normally",
    ) -> TradingSession:
        """Close the session and persist final state.

        Parameters
        ----------
        session_id : str, optional
            Session ID to close. Defaults to current active session.
        final_equity : float, optional
            Final equity value to persist.
        reason : str
            Reason for closing.

        Returns
        -------
        TradingSession
            The updated trading session record.
        """
        sid = session_id or self._state.session_id
        if sid is None:
            raise ValueError("No session to close")

        now = self._now()
        session = self.repo.get_trading_session(sid)
        if session is None:
            raise ValueError(f"Session {sid} not found")

        # Do not clear the kill switch automatically
        ks = self.repo.current_kill_switch()
        if ks is not None and ks.active:
            close_meta = {"kill_switch_remained_active": True}

        record = self.repo.update_session_status(
            sid,
            status=SessionState.CLOSED.value,
            previous_state=session.status,
            reason=reason,
            final_equity=final_equity,
        )

        # Update the in-memory state
        self._state.current_state = SessionState.CLOSED
        self._state.previous_state = SessionState(session.status)
        self._state.state_changed_at = now
        self._state.current_session = record

        # Mark ended_at in metadata
        self.repo.update_session_metadata(
            sid,
            ended_at=now.isoformat(),
            closed_at=now.isoformat(),
        )

        return record

    def mark_failed(
        self,
        session_id: str | None = None,
        *,
        failure_reason: str = "Unrecoverable error",
    ) -> TradingSession:
        """Mark the session as FAILED.

        Parameters
        ----------
        session_id : str, optional
            Session ID to mark. Defaults to current active session.
        failure_reason : str
            Description of the failure.

        Returns
        -------
        TradingSession
            The updated trading session record.
        """
        sid = session_id or self._state.session_id
        if sid is None:
            raise ValueError("No session to mark failed")

        session = self.repo.get_trading_session(sid)
        if session is None:
            raise ValueError(f"Session {sid} not found")

        record = self.repo.update_session_status(
            sid,
            status=SessionState.FAILED.value,
            previous_state=session.status,
            reason=failure_reason,
        )
        self.repo.update_session_metadata(sid, failure_reason=failure_reason)

        self._state.current_state = SessionState.FAILED
        self._state.previous_state = SessionState(session.status)
        self._state.current_session = record

        return record

    def mark_recovery_required(
        self,
        session_id: str,
        *,
        reason: str = "Resolution required",
    ) -> TradingSession:
        """Mark a session as RECOVERY_REQUIRED.

        Parameters
        ----------
        session_id : str
            Session ID to mark.
        reason : str
            Description of why recovery is required.

        Returns
        -------
        TradingSession
            The updated trading session record.
        """
        session = self.repo.get_trading_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        record = self.repo.update_session_status(
            session_id,
            status=SessionState.RECOVERY_REQUIRED.value,
            previous_state=session.status,
            reason=reason,
        )
        self.repo.update_session_metadata(session_id, recovery_reason=reason)

        self._state.current_state = SessionState.RECOVERY_REQUIRED
        self._state.previous_state = SessionState(session.status)
        self._state.current_session = record

        return record

    def next_wakeup_time(self) -> datetime | None:
        """Return the next datetime when the scheduler should be polled.

        Returns None if the session is CLOSED (no further polling needed today).
        """
        now = self._now()
        today = now.date()

        if not self._is_weekday(today) or self._is_holiday(today):
            # Next weekday that is not a holiday
            next_day = today + timedelta(days=1)
            while not self._is_weekday(next_day) or next_day in self.holidays:
                next_day += timedelta(days=1)
            return self._market_open_time(next_day)

        current_time = now.time()

        # Before market open -> wake at market open
        if current_time < self.market_start:
            return self._market_open_time(today)

        # During market -> wake at next boundary
        if current_time < self.no_new_entry_cutoff:
            return self._entry_cutoff_time(today)
        if current_time < self.squareoff_start:
            return self._squareoff_time(today)
        if current_time < self.market_end:
            return self._market_close_time(today)

        # Market closed -> wake at next market open
        next_day = today + timedelta(days=1)
        while not self._is_weekday(next_day) or next_day in self.holidays:
            next_day += timedelta(days=1)
        return self._market_open_time(next_day)

    # ------------------------------------------------------------------
    # State transition validation
    # ------------------------------------------------------------------

    def _validate_transition(self, from_state: SessionState, to_state: SessionState) -> None:
        """Raise ValueError if the state transition is disallowed."""
        allowed = ALLOWED_STATE_TRANSITIONS.get(from_state, set())
        if to_state not in allowed and from_state != to_state:
            raise ValueError(
                f"Invalid session state transition: {from_state.value} -> {to_state.value}"
            )

    # ------------------------------------------------------------------
    # Periodic poll (called by the runtime loop)
    # ------------------------------------------------------------------

    def poll(self) -> SessionState:
        """Poll the current time and update session phase if needed.

        Must be called after ``open_or_resume_session``.

        Returns the current SessionState.
        """
        if not self._state._initialized:
            raise RuntimeError("Session scheduler not initialized; call open_or_resume_session first")

        now = self._now()
        self._check_backward_clock(now)

        phase = self._determine_phase(now)
        current = self._state.current_state

        if phase != current:
            self._validate_transition(current, phase)

            # Persist the transition
            sid = self._state.session_id
            if sid is not None:
                session = self.repo.get_trading_session(sid)
                if session is not None:
                    self.repo.update_session_status(
                        sid,
                        status=phase.value,
                        previous_state=current.value,
                        expected_version=session.version,
                        reason="Phase transition via poll",
                    )

            self._state.previous_state = current
            self._state.current_state = phase
            self._state.state_changed_at = now

        self._state.last_poll_time = now
        return phase

    # ------------------------------------------------------------------
    # Validation & drift check (standalone, no side effects)
    # ------------------------------------------------------------------

    def validate_clock(
        self,
        reference_time: datetime | None = None,
        *,
        raise_on_error: bool = True,
    ) -> tuple[bool, str | None]:
        """Validate clock sanity.

        Parameters
        ----------
        reference_time : datetime, optional
            An external reference time to compare against.
        raise_on_error : bool
            If True, raise on violation. Default True.

        Returns
        -------
        (is_valid, error_message)
        """
        now = self._now()
        try:
            self._check_backward_clock(now)
            if reference_time is not None:
                self._check_clock_drift(now, reference_time)
            return True, None
        except RuntimeError as exc:
            if raise_on_error:
                raise
            return False, str(exc)