from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from runtime.end_of_day import EndOfDayController
from runtime.paper_portfolio_service import PaperPortfolioService
from runtime.position_monitor import PositionMonitor, PositionQuote
from storage.database import OperationalDatabase
from storage.repositories import OperationalRepository
from utils.time_utils import KOLKATA_TZ

NOW = datetime(2026, 7, 20, 15, 25, tzinfo=KOLKATA_TZ)


def build(tmp_path):
    db = OperationalDatabase(tmp_path / "state.db"); db.initialize()
    repo = OperationalRepository(db); repo.create_trading_session("s1", trading_date="2026-07-20")
    portfolio = PaperPortfolioService(db, initial_capital=1_000_000, session_id="s1"); portfolio.initialize()
    repo.save_position("p1", "s1", "OPT", 10, 100)
    submissions = []
    monitor = PositionMonitor(repo, portfolio, session_id="s1",
        quote_provider=lambda symbol: PositionQuote(symbol, 105, NOW),
        exit_submitter=lambda *args: submissions.append(args) or {"status": "FILLED", "order_id": "exit"},
        clock=lambda: NOW)
    controller = EndOfDayController(repo, portfolio, monitor, session_id="s1",
        reconciliation_runner=lambda: SimpleNamespace(entry_blocked=False, outcome="CLEAN"),
        cancel_order=lambda key: None, report_dir=tmp_path / "reports", dashboard_dir=tmp_path / "dashboard",
        archive_dir=tmp_path / "archive", logs_dir=tmp_path / "logs", market_data_dir=tmp_path / "market",
        clock=lambda: NOW)
    return db, repo, submissions, controller


def test_eod_closes_reconciles_persists_reports_and_is_idempotent(tmp_path):
    db, repo, submissions, controller = build(tmp_path)
    (tmp_path / "logs").mkdir(); (tmp_path / "logs" / "app.log").write_text("safe", encoding="utf-8")
    (tmp_path / "market").mkdir(); (tmp_path / "market" / "quotes.json").write_text("{}", encoding="utf-8")
    first = controller.run(); second = controller.run()
    assert first.status == second.status == "COMPLETED"
    assert len(submissions) == 1
    assert repo.get_trading_session("s1").status == "CLOSED"
    assert (tmp_path / "reports" / "s1.json").exists()
    assert (tmp_path / "dashboard" / "end_of_day.json").exists()
    assert len(first.archives) == 2
    db.dispose()


def test_eod_remains_blocked_when_exit_is_unknown(tmp_path):
    db, repo, _, controller = build(tmp_path)
    controller.position_monitor.exit_submitter = lambda *args: (_ for _ in ()).throw(TimeoutError("unknown"))
    result = controller.run()
    assert result.status == "BLOCKED" and result.positions_remaining == 1
    assert repo.get_trading_session("s1").status != "CLOSED"
    assert repo.get_trading_session("s1").metadata_json["eod"]["status"] == "BLOCKED"
    db.dispose()

