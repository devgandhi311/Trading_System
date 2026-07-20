"""Focused lifecycle tests for the autonomous PAPER engine."""

from __future__ import annotations

from datetime import datetime

from runtime.paper_trading_engine import PaperTradingEngine
from storage.database import OperationalDatabase
from storage.repositories import OperationalRepository
from utils.time_utils import KOLKATA_TZ


class _BlockedScheduler:
    def __init__(self, session) -> None:
        self.session = session
        self.poll_count = 0

    def open_or_resume_session(self):
        return self.session

    def poll(self):
        self.poll_count += 1

    def entry_allowed(self) -> bool:
        return False


def _engine(tmp_path, *, scheduler, input_provider=None) -> PaperTradingEngine:
    database = OperationalDatabase(tmp_path / "state.db")
    database.initialize()
    existing = OperationalRepository(database).get_trading_session(scheduler.session.id)
    if existing is None:
        scheduler.session = OperationalRepository(database).create_trading_session(
            scheduler.session.id, "PAPER", trading_date=scheduler.session.trading_date,
            status=scheduler.session.status,
        )
    return PaperTradingEngine(
        database=database,
        state_dir=tmp_path / "dashboard",
        scheduler=scheduler,
        input_provider=input_provider,
        clock=lambda: datetime(2026, 7, 20, 10, 0, tzinfo=KOLKATA_TZ),
        poll_interval_seconds=0.1,
    )


def test_scheduler_block_never_requests_market_data_and_projects_state(tmp_path) -> None:
    database = OperationalDatabase(tmp_path / "seed.db")
    database.initialize()
    session = OperationalRepository(database).create_trading_session("PAPER_2026-07-20", "PAPER")
    database.dispose()
    calls = 0

    def forbidden_provider():
        nonlocal calls
        calls += 1
        raise AssertionError("blocked cycles must not request market data")

    scheduler = _BlockedScheduler(session)
    engine = _engine(tmp_path, scheduler=scheduler, input_provider=forbidden_provider)
    try:
        assert engine.run_cycle() == "SCHEDULER_BLOCKED"
        assert calls == 0
        assert (tmp_path / "dashboard" / "prediction.json").exists()
        assert (tmp_path / "dashboard" / "risk.json").exists()
        assert (tmp_path / "dashboard" / "active_orders.csv").exists()
    finally:
        engine.close()


def test_continuous_mode_runs_multiple_bounded_cycles(tmp_path) -> None:
    database = OperationalDatabase(tmp_path / "seed.db")
    database.initialize()
    session = OperationalRepository(database).create_trading_session("PAPER_2026-07-20", "PAPER")
    database.dispose()
    scheduler = _BlockedScheduler(session)
    engine = _engine(tmp_path, scheduler=scheduler)
    try:
        assert engine.run(continuous=True, max_cycles=3) == 3
        assert scheduler.poll_count == 3
        engine.request_shutdown()
        assert engine.stop_event.is_set()
    finally:
        engine.close()
