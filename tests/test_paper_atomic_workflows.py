from __future__ import annotations

from datetime import datetime, timezone

import pytest

from runtime.paper_portfolio_service import PaperPortfolioService
from storage import OperationalDatabase, OperationalRepository

NOW = datetime(2026, 7, 20, 10, tzinfo=timezone.utc)


@pytest.fixture
def state(tmp_path):
    db = OperationalDatabase(tmp_path / "atomic.db"); db.initialize()
    repo = OperationalRepository(db); repo.create_trading_session("s1", started_at=NOW)
    yield db, repo
    db.dispose()


def _intent(repo):
    return repo.reserve_order_intent("key", session_id="s1", instrument_token="t", symbol="OPT",
        side="BUY", order_type="LIMIT", quantity=10, product_type="MIS", validity="DAY")


def test_fill_and_order_update_roll_back_together(state):
    _, repo = state
    order = _intent(repo)
    with pytest.raises(RuntimeError, match="injected"):
        repo.apply_fill_and_update_order(fill_id="f1", idempotency_key="key", quantity=10, price=100,
            transaction_cost=1, new_state="FILLED", expected_version=order.version, metadata={},
            filled_at=NOW, fail_before_commit=True)
    assert repo.get_order_intent("key").filled_quantity == 0
    assert repo.fills_for_order("key") == []


def test_atomic_fill_retry_is_idempotent(state):
    _, repo = state
    order = _intent(repo)
    repo.apply_fill_and_update_order(fill_id="f1", idempotency_key="key", quantity=10, price=100,
        transaction_cost=1, new_state="FILLED", expected_version=order.version, metadata={}, filled_at=NOW)
    repo.apply_fill_and_update_order(fill_id="f1", idempotency_key="key", quantity=10, price=100,
        transaction_cost=1, new_state="FILLED", expected_version=order.version, metadata={}, filled_at=NOW)
    assert len(repo.fills_for_order("key")) == 1


def test_exit_position_portfolio_and_decision_roll_back_together(state):
    db, repo = state
    PaperPortfolioService(db, initial_capital=1_000_000, session_id="s1").initialize()
    repo.save_position("p1", "s1", "OPT", 10, 100, metadata={"risk_amount": 500, "margin_required": 1000})
    repo.reserve_exit_decision(exit_id="e1", session_id="s1", position_id="p1", reason="MANUAL_EXIT")
    before = repo.latest_portfolio_for_session("s1")
    with pytest.raises(RuntimeError, match="injected"):
        repo.complete_exit_atomic(exit_id="e1", exit_price=101, realized_pnl=10,
                                  completed_at=NOW, fail_before_commit=True)
    assert repo.get_position("p1").status == "OPEN"
    assert repo.get_exit_decision("e1").metadata_json["status"] == "RESERVED"
    assert repo.latest_portfolio_for_session("s1").id == before.id
