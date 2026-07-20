from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from models.paper_models import ReconciliationOutcome
from runtime.paper_portfolio_service import PaperPortfolioService
from runtime.position_monitor import PositionMonitor, PositionQuote
from runtime.recovery_manager import RecoveryManager
from storage.database import OperationalDatabase
from storage.repositories import OperationalRepository
from utils.time_utils import KOLKATA_TZ

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=KOLKATA_TZ)


class Scheduler:
    def __init__(self, session): self.session = session
    def open_or_resume_session(self): return self.session


class Reconciliation:
    def __init__(self, outcome="CLEAN", blocked=False): self.outcome, self.blocked = outcome, blocked
    def reconcile(self, *args):
        return SimpleNamespace(outcome=ReconciliationOutcome(self.outcome), entry_blocked=self.blocked,
                               blocking_reasons=("mismatch",) if self.blocked else ())


def dependencies(tmp_path, *, outcome="CLEAN", blocked=False):
    db = OperationalDatabase(tmp_path / "state.db"); db.initialize()
    repo = OperationalRepository(db); session = repo.create_trading_session("s1", trading_date="2026-07-20")
    portfolio_factory = lambda sid: PaperPortfolioService(db, initial_capital=1_000_000, session_id=sid)
    submissions = []
    def monitor_factory(sid, portfolio):
        return PositionMonitor(repo, portfolio, session_id=sid,
            quote_provider=lambda symbol: PositionQuote(symbol, 105, NOW),
            exit_submitter=lambda *args: submissions.append(args) or {"status": "FILLED", "order_id": "exit"},
            clock=lambda: NOW)
    manager = RecoveryManager(repo, Scheduler(session), portfolio_factory,
        lambda sid: Reconciliation(outcome, blocked), monitor_factory,
        broker_order_provider=lambda: {}, broker_fill_provider=lambda: {}, broker_position_provider=lambda: {})
    return db, repo, submissions, manager


def test_clean_recovery_resumes_reserved_exit_once(tmp_path):
    db, repo, submissions, manager = dependencies(tmp_path)
    repo.save_position("p1", "s1", "OPT", 10, 100)
    repo.reserve_exit_decision(exit_id=PositionMonitor.exit_id("p1"), session_id="s1",
                               position_id="p1", reason="MANUAL_EXIT")
    first = manager.recover(); second = manager.recover()
    assert first.status == second.status == "RECOVERED"
    assert first.exits_resumed == 1 and second.exits_resumed == 0
    assert len(submissions) == 1 and repo.get_position("p1").status == "CLOSED"
    db.dispose()


def test_unknown_order_blocks_recovery_and_is_not_resubmitted(tmp_path):
    db, repo, submissions, manager = dependencies(tmp_path)
    row = repo.reserve_order_intent(idempotency_key="o1", client_order_id="o1", session_id="s1",
        instrument_token="1", symbol="OPT", side="BUY", order_type="LIMIT", quantity=1,
        product_type="MIS", validity="DAY", local_state="UNKNOWN")
    result = manager.recover()
    assert result.status == "BLOCKED"
    assert any("unknown order" in item for item in result.blockers)
    assert repo.get_order_intent("o1").id == row.id and submissions == []
    db.dispose()


def test_reconciliation_mismatch_blocks_pending_exit(tmp_path):
    db, repo, submissions, manager = dependencies(tmp_path, outcome="POSITION_MISMATCH", blocked=True)
    repo.save_position("p1", "s1", "OPT", 10, 100)
    repo.reserve_exit_decision(exit_id=PositionMonitor.exit_id("p1"), session_id="s1",
                               position_id="p1", reason="MANUAL_EXIT")
    result = manager.recover()
    assert result.status == "BLOCKED" and result.exits_resumed == 0 and submissions == []
    db.dispose()
