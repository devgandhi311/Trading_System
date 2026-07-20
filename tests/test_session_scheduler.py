"""Deterministic tests for the durable PAPER session scheduler and lifecycle controller.

Tests cover all 24 required scenarios plus edge cases.
"""

from __future__ import annotations

import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Generator
from zoneinfo import ZoneInfo

import pytest

from models.paper_models import SessionState
from storage.database import OperationalDatabase
from storage.models import (
    KillSwitchEvent,
    OrderIntentRecord,
    PositionRecord,
    ReconciliationEvent,
    TradingSession,
)
from storage.repositories import OperationalRepository, OptimisticLockError
from runtime.session_scheduler import SessionScheduler

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_clock(dt: datetime) -> Callable[[], datetime]:
    """Return a clock function that always returns the given datetime."""
    def _clock() -> datetime:
        return dt
    return _clock


def advance_clock(base: datetime, **delta: Any) -> Callable[[], datetime]:
    """Return a clock that returns base + timedelta(**delta)."""
    target = base + timedelta(**delta)
    return make_clock(target)


def make_ist(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    """Create an Asia/Kolkata-aware datetime."""
    return datetime(year, month, day, hour, minute, second, tzinfo=KOLKATA_TZ)


# Standard market times
MARKET_OPEN = time(9, 15)
ENTRY_CUTOFF = time(15, 0)
SQUAREOFF_START = time(15, 20)
MARKET_CLOSE = time(15, 30)

# A Monday
MONDAY = make_ist(2026, 7, 13)
# A Tuesday
TUESDAY = make_ist(2026, 7, 14)
# A Wednesday
WEDNESDAY = make_ist(2026, 7, 15)
# A Thursday
THURSDAY = make_ist(2026, 7, 16)
# A Friday
FRIDAY = make_ist(2026, 7, 17)
# A Saturday
SATURDAY = make_ist(2026, 7, 18)
# A Sunday
SUNDAY = make_ist(2026, 7, 19)
# A Monday next week
MONDAY_NEXT = make_ist(2026, 7, 20)

# A configured holiday - use a Friday that is a working day normally
HOLIDAY = date(2026, 7, 17)  # Friday, configured as holiday


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary database file."""
    db_file = tmp_path / "test_session_scheduler.db"
    yield db_file
    try:
        if db_file.exists():
            db_file.unlink()
    except PermissionError:
        pass


@pytest.fixture
def database(db_path: Path) -> Generator[OperationalDatabase, None, None]:
    """Create and initialize an operational database."""
    db = OperationalDatabase(db_path)
    db.initialize()
    yield db
    db.dispose()


@pytest.fixture
def repo(database: OperationalDatabase) -> OperationalRepository:
    return OperationalRepository(database)


# ---------------------------------------------------------------------------
# 1. Preopen phase
# ---------------------------------------------------------------------------


class TestPreopenPhase:
    """Session must be PREOPEN before market open on a weekday."""

    def test_preopen_before_market_open(self, database: OperationalDatabase) -> None:
        """Before 09:15 on a weekday, phase must be PREOPEN."""
        clock = make_clock(make_ist(2026, 7, 15, 9, 0, 0))  # Wednesday 09:00
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.PREOPEN.value
        assert scheduler.get_current_phase() == SessionState.PREOPEN

    def test_preopen_at_midnight(self, database: OperationalDatabase) -> None:
        """At 00:00 on a weekday, phase must be PREOPEN."""
        clock = make_clock(make_ist(2026, 7, 15, 0, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.PREOPEN.value


# ---------------------------------------------------------------------------
# 2. Market open phase
# ---------------------------------------------------------------------------


class TestMarketOpenPhase:
    """Session must be ENTRY_ALLOWED during market hours before cutoff."""

    def test_market_open_entry_allowed(self, database: OperationalDatabase) -> None:
        """At 10:00 on a weekday, phase must be ENTRY_ALLOWED."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.ENTRY_ALLOWED.value


# ---------------------------------------------------------------------------
# 3. Entry-allowed phase
# ---------------------------------------------------------------------------


class TestEntryAllowed:
    """New entries must be permitted during ENTRY_ALLOWED."""

    def test_entry_allowed_returns_true(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.entry_allowed() is True

    def test_entry_allowed_exits_allowed(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.exits_allowed() is True


# ---------------------------------------------------------------------------
# 4. Entry cutoff
# ---------------------------------------------------------------------------


class TestEntryCutoff:
    """After entry cutoff, new entries must be blocked."""

    def test_entry_blocked_after_cutoff(self, database: OperationalDatabase) -> None:
        """At 15:05, entry must be blocked."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 5, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.entry_allowed() is False

    def test_entry_blocked_at_cutoff_exact(self, database: OperationalDatabase) -> None:
        """At exactly 15:00, entry must still be allowed (inclusive)."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.entry_allowed() is True


# ---------------------------------------------------------------------------
# 5. EXIT_ONLY phase
# ---------------------------------------------------------------------------


class TestExitOnly:
    """After entry cutoff but before square-off, only exits allowed."""

    def test_exit_only_phase(self, database: OperationalDatabase) -> None:
        """At 15:10, phase must be EXIT_ONLY."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 10, 0))
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.EXIT_ONLY.value
        assert scheduler.entry_allowed() is False
        assert scheduler.exits_allowed() is True
        assert scheduler.squareoff_required() is True


# ---------------------------------------------------------------------------
# 6. SQUAREOFF phase
# ---------------------------------------------------------------------------


class TestSquareoff:
    """During square-off window, positions must be closed."""

    def test_squareoff_phase(self, database: OperationalDatabase) -> None:
        """At 15:25, phase must be SQUAREOFF."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 25, 0))
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.SQUAREOFF.value
        assert scheduler.entry_allowed() is False
        assert scheduler.exits_allowed() is True
        assert scheduler.squareoff_required() is True


# ---------------------------------------------------------------------------
# 7. Market close
# ---------------------------------------------------------------------------


class TestMarketClose:
    """After market close, session must be CLOSED."""

    def test_market_close_phase(self, database: OperationalDatabase) -> None:
        """At 15:35, phase must be CLOSED."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 35, 0))
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.CLOSED.value


# ---------------------------------------------------------------------------
# 8. Weekend
# ---------------------------------------------------------------------------


class TestWeekend:
    """Weekends must be detected as HOLIDAY."""

    def test_saturday_is_holiday(self, database: OperationalDatabase) -> None:
        clock = make_clock(SATURDAY)
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.HOLIDAY.value

    def test_sunday_is_holiday(self, database: OperationalDatabase) -> None:
        clock = make_clock(SUNDAY)
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.HOLIDAY.value


# ---------------------------------------------------------------------------
# 9. Configured holiday
# ---------------------------------------------------------------------------


class TestConfiguredHoliday:
    """Configured NSE holidays must be detected."""

    def test_configured_holiday(self, database: OperationalDatabase) -> None:
        """Republic Day (Jan 26) must be a configured weekday holiday."""
        holiday_date = make_ist(2026, 1, 26, 10, 0, 0)
        # Verify it's a weekday
        assert holiday_date.weekday() < 5
        clock = make_clock(holiday_date)
        scheduler = SessionScheduler(
            database, clock=clock, holidays={date(2026, 1, 26)}
        )
        session = scheduler.open_or_resume_session()
        assert session.status == SessionState.HOLIDAY.value


# ---------------------------------------------------------------------------
# 10. Restart during active session
# ---------------------------------------------------------------------------


class TestRestartDuringActiveSession:
    """Restart must resume an existing active session."""

    def test_restart_resumes_active_session(self, database: OperationalDatabase) -> None:
        """Create a session, then simulate restart by creating a new scheduler."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler1 = SessionScheduler(database, clock=clock)
        session1 = scheduler1.open_or_resume_session()
        assert session1.status == SessionState.ENTRY_ALLOWED.value

        # Simulate restart: new scheduler instance
        scheduler2 = SessionScheduler(database, clock=clock)
        session2 = scheduler2.open_or_resume_session()
        assert session2.id == session1.id
        assert session2.status == SessionState.ENTRY_ALLOWED.value
        assert scheduler2.get_current_phase() == SessionState.ENTRY_ALLOWED


# ---------------------------------------------------------------------------
# 11. Duplicate-session prevention
# ---------------------------------------------------------------------------


class TestDuplicateSessionPrevention:
    """Two attempts to open the same session must return one authoritative session."""

    def test_duplicate_session_returns_same(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        session1 = scheduler.open_or_resume_session()
        session2 = scheduler.open_or_resume_session()
        assert session1.id == session2.id

    def test_duplicate_session_different_scheduler(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler1 = SessionScheduler(database, clock=clock)
        scheduler2 = SessionScheduler(database, clock=clock)
        session1 = scheduler1.open_or_resume_session()
        session2 = scheduler2.open_or_resume_session()
        assert session1.id == session2.id


# ---------------------------------------------------------------------------
# 12. Concurrent open attempts
# ---------------------------------------------------------------------------


class TestConcurrentOpenAttempts:
    """Concurrent open attempts must produce one authoritative session."""

    def test_concurrent_open(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))

        def open_session(_: int) -> str:
            scheduler = SessionScheduler(database, clock=clock)
            session = scheduler.open_or_resume_session()
            return session.id

        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(open_session, range(8)))
        assert len(set(ids)) == 1


# ---------------------------------------------------------------------------
# 13. Backward clock movement
# ---------------------------------------------------------------------------


class TestBackwardClock:
    """Backward clock movement must be detected."""

    def test_backward_clock_detected(self, database: OperationalDatabase) -> None:
        """Moving clock backward between polls must raise RuntimeError."""
        times = iter([
            make_ist(2026, 7, 15, 10, 0, 0),  # first poll
            make_ist(2026, 7, 15, 9, 59, 0),  # backward!
        ])
        clock = lambda: next(times)
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        with pytest.raises(RuntimeError, match="Backward clock"):
            scheduler.poll()


# ---------------------------------------------------------------------------
# 14. Clock drift violation
# ---------------------------------------------------------------------------


class TestClockDrift:
    """Excessive clock drift from reference time must be detected."""

    def test_clock_drift_violation(self, database: OperationalDatabase) -> None:
        """Clock drift > 60s from reference must raise RuntimeError."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock, max_drift_seconds=60.0)
        reference = make_ist(2026, 7, 15, 10, 2, 0)  # 2 minutes ahead
        with pytest.raises(RuntimeError, match="Clock drift"):
            scheduler.validate_clock(reference)

    def test_clock_drift_within_limits(self, database: OperationalDatabase) -> None:
        """Clock drift within limits must pass."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock, max_drift_seconds=60.0)
        reference = make_ist(2026, 7, 15, 10, 0, 30)  # 30 seconds ahead
        valid, _ = scheduler.validate_clock(reference, raise_on_error=False)
        assert valid is True


# ---------------------------------------------------------------------------
# 15. Unresolved previous session
# ---------------------------------------------------------------------------


class TestUnresolvedPreviousSession:
    """A prior unresolved session must cause RECOVERY_REQUIRED."""

    def test_unresolved_previous_session(self, database: OperationalDatabase, repo: OperationalRepository) -> None:
        """Create an unresolved session from yesterday, then try to open today."""
        yesterday = make_ist(2026, 7, 14, 10, 0, 0)
        today = make_ist(2026, 7, 15, 10, 0, 0)

        # Create yesterday's session manually (unresolved - still OPEN)
        repo.create_trading_session(
            session_id="PAPER_2026-07-14",
            mode="PAPER",
            trading_date="2026-07-14",
            status="OPEN",
        )

        # Try to open today's session
        clock = make_clock(today)
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        # The unresolved previous session should cause RECOVERY_REQUIRED
        assert session.status == SessionState.RECOVERY_REQUIRED.value


# ---------------------------------------------------------------------------
# 16. UNKNOWN order causes RECOVERY_REQUIRED
# ---------------------------------------------------------------------------


class TestUnknownOrderCausesRecovery:
    """A session with UNKNOWN orders must not start cleanly."""

    def test_unknown_order_blocks_clean_start(self, database: OperationalDatabase, repo: OperationalRepository) -> None:
        """Create a session with an UNKNOWN order, then try to open a new session."""
        yesterday = make_ist(2026, 7, 14, 10, 0, 0)
        today = make_ist(2026, 7, 15, 10, 0, 0)

        # Create yesterday's session with an UNKNOWN order
        repo.create_trading_session(
            session_id="PAPER_2026-07-14",
            mode="PAPER",
            trading_date="2026-07-14",
            status="OPEN",
        )
        # Add an UNKNOWN order
        repo.reserve_order_intent(
            idempotency_key="unknown-order-1",
            session_id="PAPER_2026-07-14",
            instrument_token="26000",
            symbol="NIFTY",
            side="BUY",
            order_type="LIMIT",
            quantity=75,
            limit_price=100.0,
            product_type="MIS",
            validity="DAY",
            local_state="UNKNOWN",
        )

        # Try to open today's session
        clock = make_clock(today)
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        # The unresolved previous session with UNKNOWN order should cause RECOVERY_REQUIRED
        assert session.status == SessionState.RECOVERY_REQUIRED.value


# ---------------------------------------------------------------------------
# 17. Reconciliation mismatch causes RECOVERY_REQUIRED
# ---------------------------------------------------------------------------


class TestReconciliationMismatchCausesRecovery:
    """A session with unresolved reconciliation must not start cleanly."""

    def test_reconciliation_mismatch_blocks_clean_start(self, database: OperationalDatabase, repo: OperationalRepository) -> None:
        """Create a session with a blocking reconciliation, then try to open a new session."""
        yesterday = make_ist(2026, 7, 14, 10, 0, 0)
        today = make_ist(2026, 7, 15, 10, 0, 0)

        # Create yesterday's session with a blocking reconciliation
        repo.create_trading_session(
            session_id="PAPER_2026-07-14",
            mode="PAPER",
            trading_date="2026-07-14",
            status="OPEN",
        )
        repo.save_reconciliation_event(
            id="recon-mismatch-1",
            session_id="PAPER_2026-07-14",
            outcome="QUANTITY_MISMATCH",
            entry_blocked=True,
            exits_allowed=True,
            occurred_at=yesterday,
            reason="Quantity mismatch",
            affected_order_ids=[],
            affected_position_ids=[],
            local_state={},
            broker_state={},
            mismatch_details=[],
            matched_orders=0,
            matched_fills=0,
            matched_positions=0,
            repaired_items=[],
            blocking_reasons=["Quantity mismatch"],
            warnings=[],
            detail={},
        )

        # Try to open today's session
        clock = make_clock(today)
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        # The unresolved previous session with blocking reconciliation should cause RECOVERY_REQUIRED
        assert session.status == SessionState.RECOVERY_REQUIRED.value


# ---------------------------------------------------------------------------
# 18. Existing exits allowed after entry cutoff
# ---------------------------------------------------------------------------


class TestExitsAllowedAfterCutoff:
    """Exits must be allowed after entry cutoff."""

    def test_exits_allowed_in_exit_only(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 15, 10, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.exits_allowed() is True

    def test_exits_allowed_in_squareoff(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 15, 25, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.exits_allowed() is True


# ---------------------------------------------------------------------------
# 19. New entry blocked after cutoff
# ---------------------------------------------------------------------------


class TestNewEntryBlockedAfterCutoff:
    """New entries must be blocked after entry cutoff."""

    def test_entry_blocked_in_exit_only(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 15, 10, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.entry_allowed() is False

    def test_entry_blocked_in_squareoff(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 15, 25, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.entry_allowed() is False

    def test_entry_blocked_in_closed(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 15, 35, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.entry_allowed() is False


# ---------------------------------------------------------------------------
# 20. State-transition persistence
# ---------------------------------------------------------------------------


class TestStateTransitionPersistence:
    """Every state transition must be persisted."""

    def test_transition_persisted(self, database: OperationalDatabase) -> None:
        """Polling through a phase change must persist the transition."""
        # Start at 09:00 (PREOPEN)
        times = iter([
            make_ist(2026, 7, 15, 9, 0, 0),   # PREOPEN
            make_ist(2026, 7, 15, 10, 0, 0),  # ENTRY_ALLOWED
        ])
        clock = lambda: next(times)
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.get_current_phase() == SessionState.PREOPEN

        # Poll to trigger transition
        phase = scheduler.poll()
        assert phase == SessionState.ENTRY_ALLOWED

        sid = scheduler._state.session_id
        assert sid is not None
        # Verify persistence
        session = scheduler.repo.get_trading_session(sid)
        assert session is not None
        assert session.status == SessionState.ENTRY_ALLOWED.value

        # Verify transition record
        transitions = scheduler.repo.get_session_transitions(sid)
        assert len(transitions) >= 2  # initial + transition


# ---------------------------------------------------------------------------
# 21. Invalid transition rejected
# ---------------------------------------------------------------------------


class TestInvalidTransition:
    """Invalid state transitions must be rejected."""

    def test_invalid_transition_raises(self, database: OperationalDatabase) -> None:
        """Transition from CLOSED to ENTRY_ALLOWED must fail."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 35, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.get_current_phase() == SessionState.CLOSED

        # Try to poll at a time that would cause CLOSED -> ENTRY_ALLOWED
        # This should not happen in practice since time only moves forward,
        # but the validation should catch it.
        scheduler._state.current_state = SessionState.CLOSED
        with pytest.raises(ValueError, match="Invalid session state transition"):
            scheduler._validate_transition(SessionState.CLOSED, SessionState.ENTRY_ALLOWED)


# ---------------------------------------------------------------------------
# 22. Final equity reference retained on close
# ---------------------------------------------------------------------------


class TestFinalEquityRetained:
    """Final equity must be retained when closing a session."""

    def test_final_equity_retained(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()

        # Close with final equity
        scheduler.close_session(final_equity=987654.32)

        sid = scheduler._state.session_id
        assert sid is not None
        # Verify persistence
        session = scheduler.repo.get_trading_session(sid)
        assert session is not None
        # Check metadata for final_equity
        meta = session.metadata_json
        assert meta.get("final_equity") == 987654.32


# ---------------------------------------------------------------------------
# 23. Active kill switch is not cleared
# ---------------------------------------------------------------------------


class TestKillSwitchNotCleared:
    """Closing a session must not clear the kill switch."""

    def test_kill_switch_not_cleared_on_close(self, database: OperationalDatabase) -> None:
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()

        # Trigger kill switch
        scheduler.repo.trigger_kill_switch(reason="Test kill", source="test")

        # Close session
        scheduler.close_session()

        # Kill switch must still be active
        ks = scheduler.repo.current_kill_switch()
        assert ks is not None
        assert ks.active is True


# ---------------------------------------------------------------------------
# 24. Next wake-up calculation
# ---------------------------------------------------------------------------


class TestNextWakeup:
    """Next wake-up time must be calculated correctly."""

    def test_wakeup_before_market_open(self, database: OperationalDatabase) -> None:
        """Before market open, wakeup should be at market open."""
        clock = make_clock(make_ist(2026, 7, 15, 8, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        wakeup = scheduler.next_wakeup_time()
        expected = make_ist(2026, 7, 15, 9, 15, 0)
        assert wakeup == expected

    def test_wakeup_during_entry_allowed(self, database: OperationalDatabase) -> None:
        """During entry allowed, wakeup should be at entry cutoff."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        wakeup = scheduler.next_wakeup_time()
        expected = make_ist(2026, 7, 15, 15, 0, 0)
        assert wakeup == expected

    def test_wakeup_during_exit_only(self, database: OperationalDatabase) -> None:
        """During exit only, wakeup should be at squareoff start."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 10, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        wakeup = scheduler.next_wakeup_time()
        expected = make_ist(2026, 7, 15, 15, 20, 0)
        assert wakeup == expected

    def test_wakeup_during_squareoff(self, database: OperationalDatabase) -> None:
        """During squareoff, wakeup should be at market close."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 25, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        wakeup = scheduler.next_wakeup_time()
        expected = make_ist(2026, 7, 15, 15, 30, 0)
        assert wakeup == expected

    def test_wakeup_after_market_close(self, database: OperationalDatabase) -> None:
        """After market close, wakeup should be at next market open."""
        clock = make_clock(make_ist(2026, 7, 15, 15, 35, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        wakeup = scheduler.next_wakeup_time()
        expected = make_ist(2026, 7, 16, 9, 15, 0)  # Next day (Thursday)
        assert wakeup == expected

    def test_wakeup_on_friday_after_close(self, database: OperationalDatabase) -> None:
        """After Friday close, wakeup should be Monday market open."""
        clock = make_clock(make_ist(2026, 7, 17, 15, 35, 0))  # Friday
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        wakeup = scheduler.next_wakeup_time()
        expected = make_ist(2026, 7, 20, 9, 15, 0)  # Monday
        assert wakeup == expected

    def test_wakeup_on_weekend(self, database: OperationalDatabase) -> None:
        """On weekend, wakeup should be Monday market open."""
        clock = make_clock(SATURDAY)
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        wakeup = scheduler.next_wakeup_time()
        expected = make_ist(2026, 7, 20, 9, 15, 0)  # Monday
        assert wakeup == expected

    def test_wakeup_before_holiday(self, database: OperationalDatabase) -> None:
        """Day before holiday, wakeup should skip to next non-holiday."""
        # Aug 14 (Friday) before Independence Day (Aug 15, Saturday)
        # Next market day should be Aug 17 (Monday)
        clock = make_clock(make_ist(2026, 8, 14, 15, 35, 0))
        scheduler = SessionScheduler(
            database, clock=clock, holidays={date(2026, 8, 15)}
        )
        scheduler.open_or_resume_session()
        wakeup = scheduler.next_wakeup_time()
        expected = make_ist(2026, 8, 17, 9, 15, 0)  # Monday
        assert wakeup == expected


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge case tests."""

    def test_poll_without_initialization_raises(self, database: OperationalDatabase) -> None:
        """Poll without calling open_or_resume_session must raise."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        with pytest.raises(RuntimeError, match="not initialized"):
            scheduler.poll()

    def test_close_without_session_raises(self, database: OperationalDatabase) -> None:
        """Close without a session must raise."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        with pytest.raises(ValueError, match="No session to close"):
            scheduler.close_session()

    def test_mark_failed_without_session_raises(self, database: OperationalDatabase) -> None:
        """Mark failed without a session must raise."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        with pytest.raises(ValueError, match="No session to mark failed"):
            scheduler.mark_failed()

    def test_entry_allowed_before_initialization(self, database: OperationalDatabase) -> None:
        """entry_allowed before initialization must return False."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        assert scheduler.entry_allowed() is False

    def test_exits_allowed_before_initialization(self, database: OperationalDatabase) -> None:
        """exits_allowed before initialization must return False."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        assert scheduler.exits_allowed() is False

    def test_squareoff_required_before_initialization(self, database: OperationalDatabase) -> None:
        """squareoff_required before initialization must return False."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        assert scheduler.squareoff_required() is False

    def test_mark_failed_persists_reason(self, database: OperationalDatabase) -> None:
        """Marking a session as failed must persist the failure reason."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        scheduler.mark_failed(failure_reason="Connection lost")
        sid = scheduler._state.session_id
        assert sid is not None
        session = scheduler.repo.get_trading_session(sid)
        assert session is not None
        assert session.status == SessionState.FAILED.value
        meta = session.metadata_json
        assert meta.get("failure_reason") == "Connection lost"

    def test_mark_recovery_required_persists_reason(self, database: OperationalDatabase, repo: OperationalRepository) -> None:
        """Marking a session as recovery required must persist the reason."""
        # Create a session first
        repo.create_trading_session(
            session_id="PAPER_2026-07-14",
            mode="PAPER",
            trading_date="2026-07-14",
            status="OPEN",
        )
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.mark_recovery_required(
            "PAPER_2026-07-14",
            reason="Unresolved orders",
        )
        session = scheduler.repo.get_trading_session("PAPER_2026-07-14")
        assert session is not None
        assert session.status == SessionState.RECOVERY_REQUIRED.value
        meta = session.metadata_json
        assert meta.get("recovery_reason") == "Unresolved orders"

    def test_kill_switch_blocks_entry(self, database: OperationalDatabase) -> None:
        """Active kill switch must block entry even during ENTRY_ALLOWED."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.entry_allowed() is True

        # Trigger kill switch
        scheduler.repo.trigger_kill_switch(reason="Test block", source="test")
        assert scheduler.entry_allowed() is False

    def test_phase_transition_via_poll(self, database: OperationalDatabase) -> None:
        """Poll must transition through phases correctly."""
        times = iter([
            make_ist(2026, 7, 15, 9, 0, 0),   # PREOPEN
            make_ist(2026, 7, 15, 10, 0, 0),  # ENTRY_ALLOWED
            make_ist(2026, 7, 15, 15, 5, 0),  # EXIT_ONLY
            make_ist(2026, 7, 15, 15, 25, 0), # SQUAREOFF
            make_ist(2026, 7, 15, 15, 35, 0), # CLOSED
        ])
        clock = lambda: next(times)
        scheduler = SessionScheduler(database, clock=clock)
        scheduler.open_or_resume_session()
        assert scheduler.get_current_phase() == SessionState.PREOPEN

        assert scheduler.poll() == SessionState.ENTRY_ALLOWED
        assert scheduler.poll() == SessionState.EXIT_ONLY
        assert scheduler.poll() == SessionState.SQUAREOFF
        assert scheduler.poll() == SessionState.CLOSED

    def test_session_id_format(self, database: OperationalDatabase) -> None:
        """Session ID must follow the expected format."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler = SessionScheduler(database, clock=clock)
        session = scheduler.open_or_resume_session()
        assert session.id == "PAPER_2026-07-15"

    def test_different_mode_session(self, database: OperationalDatabase) -> None:
        """Sessions with different modes must not conflict."""
        clock = make_clock(make_ist(2026, 7, 15, 10, 0, 0))
        scheduler_paper = SessionScheduler(database, mode="PAPER", clock=clock)
        scheduler_live = SessionScheduler(database, mode="LIVE", clock=clock)
        session_paper = scheduler_paper.open_or_resume_session()
        session_live = scheduler_live.open_or_resume_session()
        assert session_paper.id != session_live.id
        assert session_paper.mode == "PAPER"
        assert session_live.mode == "LIVE"
