from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from models.paper_models import SessionState
from runtime.paper_qualification import PaperQualificationMode
from storage.database import OperationalDatabase
from storage.repositories import OperationalRepository
from utils.config_loader import ConfigLoader
from utils.time_utils import KOLKATA_TZ

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=KOLKATA_TZ)


class Engine:
    def __init__(self, repo, action="NO_TRADE"):
        from threading import Event
        self.repository = repo; self.config = ConfigLoader().load(); self.stop_event = Event()
        self.poll_interval_seconds = 0.01; self.action = action; self.cycles = 0
        self.scheduler = SimpleNamespace(get_current_phase=lambda: SessionState.ENTRY_ALLOWED)
    def run_cycle(self): self.cycles += 1; return self.action
    def request_shutdown(self): self.stop_event.set()


def setup(tmp_path):
    db = OperationalDatabase(tmp_path / "state.db"); db.initialize(); repo = OperationalRepository(db)
    repo.create_trading_session("s1", trading_date="2026-07-20", started_at=NOW,
        metadata={"qualification_cycle_observations": [
            {"latency_ms": 10, "duplicates_prevented": 1}, {"latency_ms": 20, "duplicates_prevented": 0}]})
    repo.save_portfolio_snapshot("s1", 1_000_000, 990_000, realized_pnl=-10_000)
    return db, repo


def test_metrics_and_daily_weekly_monthly_reports(tmp_path):
    db, repo = setup(tmp_path); mode = PaperQualificationMode(Engine(repo), report_dir=tmp_path / "reports", clock=lambda: NOW)
    metrics = mode.metrics(NOW.date(), NOW.date()); paths = mode.generate_reports()
    assert metrics["latency_ms"]["mean"] == 15
    assert metrics["duplicate_attempts_prevented"] == 1
    assert metrics["pnl"]["realized"] == -10_000
    assert set(paths) == {"daily", "weekly", "monthly"}
    assert all(path.exists() for path in paths.values())
    assert not list((tmp_path / "reports").rglob("*.tmp"))
    db.dispose()


def test_qualification_runs_multiple_cycles(tmp_path):
    db, repo = setup(tmp_path); engine = Engine(repo)
    result = PaperQualificationMode(engine, report_dir=tmp_path / "reports", clock=lambda: NOW).run(max_cycles=3)
    assert result.status == "COMPLETED" and result.cycles == 3
    db.dispose()


def test_safety_failure_stops_and_persists_kill_switch(tmp_path):
    db, repo = setup(tmp_path); repo.trigger_kill_switch(reason="test", source="test", session_id="s1")
    engine = Engine(repo); result = PaperQualificationMode(engine, report_dir=tmp_path / "reports", clock=lambda: NOW).run(max_cycles=3)
    assert result.status == "SAFETY_STOP" and result.cycles == 1
    assert engine.stop_event.is_set() and repo.current_kill_switch().active
    db.dispose()


def test_live_flags_are_never_enabled(tmp_path):
    db, repo = setup(tmp_path); mode = PaperQualificationMode(Engine(repo), report_dir=tmp_path / "reports", clock=lambda: NOW)
    report = mode.metrics(NOW.date(), NOW.date())
    assert report["mode"] == "PAPER" and report["live_order_submission"] is False
    db.dispose()


def test_cycle_exception_stops_qualification_and_still_writes_reports(tmp_path):
    db, repo = setup(tmp_path); engine = Engine(repo)
    engine.run_cycle = lambda: (_ for _ in ()).throw(RuntimeError("feed unavailable"))
    result = PaperQualificationMode(engine, report_dir=tmp_path / "reports", clock=lambda: NOW).run()
    assert result.status == "SAFETY_STOP" and "feed unavailable" in result.safety_failures[0]
    assert engine.stop_event.is_set()
    assert (tmp_path / "reports" / "daily" / "2026-07-20.json").exists()
    db.dispose()
