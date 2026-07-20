from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from runtime.paper_portfolio_service import PaperPortfolioService
from runtime.position_monitor import PositionMonitor, PositionQuote
from storage.database import OperationalDatabase
from storage.repositories import OperationalRepository
from utils.time_utils import KOLKATA_TZ

NOW = datetime(2026, 7, 20, 14, 0, tzinfo=KOLKATA_TZ)


@pytest.fixture
def setup(tmp_path):
    db = OperationalDatabase(tmp_path / "state.db"); db.initialize()
    repo = OperationalRepository(db); repo.create_trading_session("s1", trading_date="2026-07-20")
    portfolio = PaperPortfolioService(db, initial_capital=1_000_000, session_id="s1"); portfolio.initialize()
    yield db, repo, portfolio
    db.dispose()


def monitor(repo, portfolio, price, submitted, *, now=NOW):
    return PositionMonitor(repo, portfolio, session_id="s1",
                           quote_provider=lambda symbol: PositionQuote(symbol, price, now),
                           exit_submitter=submitted, clock=lambda: now)


@pytest.mark.parametrize(("metadata", "price", "reason"), [
    ({"stop_loss": 90}, 89, "STOP_LOSS"),
    ({"target": 110}, 111, "TARGET"),
    ({"high_watermark": 120, "trailing_stop_percent": 10}, 107, "TRAILING_STOP"),
    ({"time_exit_at": (NOW - timedelta(minutes=1)).isoformat()}, 101, "TIME_EXIT"),
    ({"expiry": NOW.date().isoformat()}, 101, "EXPIRY_EXIT"),
    ({"manual_exit": True}, 101, "MANUAL_EXIT"),
    ({"emergency_exit": True}, 101, "EMERGENCY_EXIT"),
])
def test_all_exit_triggers_are_persisted_before_submission(setup, metadata, price, reason):
    _, repo, portfolio = setup
    repo.save_position("p1", "s1", "OPT", 10, 100, metadata=metadata)
    observed = []
    def submit(position, exit_id, actual_reason, emergency):
        observed.append((repo.get_exit_decision(exit_id).metadata_json["status"], actual_reason, emergency))
        return {"status": "FILLED", "order_id": "x1"}
    result = monitor(repo, portfolio, price, submit).monitor_once()
    assert result.exits_completed == 1
    assert observed[0][0] == "RESERVED" and observed[0][1] == reason
    assert repo.get_position("p1").status == "CLOSED"
    assert repo.get_exit_decision(PositionMonitor.exit_id("p1")).metadata_json["status"] == "COMPLETED"


def test_duplicate_exit_is_not_submitted_after_restart(setup):
    db, repo, portfolio = setup
    repo.save_position("p1", "s1", "OPT", 10, 100, metadata={"stop_loss": 90})
    calls = []
    submit = lambda *args: calls.append(args) or {"status": "FILLED", "order_id": "x"}
    monitor(repo, portfolio, 80, submit).monitor_once()
    restarted = OperationalRepository(db)
    monitor(restarted, PaperPortfolioService(db, session_id="s1"), 80, submit).monitor_once()
    assert len(calls) == 1


def test_submit_exception_becomes_unknown_and_is_never_blindly_retried(setup):
    _, repo, portfolio = setup
    repo.save_position("p1", "s1", "OPT", 10, 100, metadata={"manual_exit": True})
    calls = 0
    def fail(*args):
        nonlocal calls; calls += 1; raise TimeoutError("ambiguous")
    instance = monitor(repo, portfolio, 100, fail)
    instance.monitor_once(); instance.monitor_once()
    assert calls == 1
    assert repo.get_exit_decision(PositionMonitor.exit_id("p1")).metadata_json["status"] == "UNKNOWN"


def test_stale_quote_fails_closed_without_exit(setup):
    _, repo, portfolio = setup
    repo.save_position("p1", "s1", "OPT", 10, 100, metadata={"stop_loss": 90})
    calls = []
    instance = PositionMonitor(repo, portfolio, session_id="s1",
        quote_provider=lambda symbol: PositionQuote(symbol, 80, NOW - timedelta(seconds=10)),
        exit_submitter=lambda *args: calls.append(args), clock=lambda: NOW)
    result = instance.monitor_once()
    assert calls == [] and "stale quote" in result.blocked[0]


def test_monitor_updates_unrealized_and_trailing_watermark(setup):
    _, repo, portfolio = setup
    repo.save_position("p1", "s1", "OPT", 10, 100, metadata={"trailing_stop_percent": 10})
    monitor(repo, portfolio, 115, lambda *args: None).monitor_once()
    position = repo.get_position("p1")
    assert position.unrealized_pnl == 150
    assert position.metadata_json["high_watermark"] == 115

