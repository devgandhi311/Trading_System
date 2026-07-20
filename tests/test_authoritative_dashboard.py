from __future__ import annotations

from datetime import datetime
import json

from runtime.dashboard_projection import AuthoritativeDashboardProjector
from storage.database import OperationalDatabase
from storage.repositories import OperationalRepository
from utils.time_utils import KOLKATA_TZ

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=KOLKATA_TZ)


def test_projection_contains_only_database_values_and_atomic_files(tmp_path):
    db = OperationalDatabase(tmp_path / "state.db"); db.initialize()
    repo = OperationalRepository(db)
    repo.create_trading_session("s1", trading_date="2026-07-20", started_at=NOW)
    repo.save_portfolio_snapshot("s1", 1_000_000, 999_500, realized_pnl=-500)
    repo.save_position("p1", "s1", "NIFTYOPT", 75, 100)
    state = AuthoritativeDashboardProjector(repo, tmp_path / "live", clock=lambda: NOW).project("s1", action="NO_TRADE")
    assert state["risk"]["current_equity"] == 999_500
    assert state["positions"][0]["symbol"] == "NIFTYOPT"
    assert state["prediction"]["available"] is False
    assert "mock" not in json.dumps(state, default=str).lower()
    assert not list((tmp_path / "live").glob("*.tmp"))
    assert json.loads((tmp_path / "live" / "dashboard_state.json").read_text())["session_id"] == "s1"
    db.dispose()


def test_projection_replaces_prior_file_without_partial_json(tmp_path):
    db = OperationalDatabase(tmp_path / "state.db"); db.initialize(); repo = OperationalRepository(db)
    repo.create_trading_session("s1", trading_date="2026-07-20", started_at=NOW)
    projector = AuthoritativeDashboardProjector(repo, tmp_path / "live", clock=lambda: NOW)
    projector.project("s1", action="FIRST"); projector.project("s1", action="SECOND")
    payload = json.loads((tmp_path / "live" / "dashboard_state.json").read_text())
    assert payload["engine_action"] == "SECOND"
    assert not list((tmp_path / "live").glob(".*.tmp"))
    db.dispose()
