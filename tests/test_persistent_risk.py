"""Comprehensive tests for persistent PAPER portfolio and risk-state service.

Tests cover:
- restart with active kill switch
- authorized clear
- unauthorized clear
- concurrent trigger
- equity after profit
- equity after loss
- costs reduce equity
- slippage reduces equity
- pending-order risk
- partial-fill risk
- margin reservation
- peak equity
- drawdown
- daily counter reset
- unresolved state survives daily reset
- zero equity
- negative equity
- optimistic-lock conflict
- P&L realized once
- reconciliation identity
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest

from storage.database import OperationalDatabase
from storage.repositories import (
    OptimisticLockError,
    PortfolioAccountingRepository,
)
from runtime.paper_portfolio_service import PaperPortfolioService
from risk.kill_switch import KillSwitch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_CAPITAL = 1_000_000.0


@pytest.fixture
def db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for each test."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="persistent_risk_"))
    db_file = tmp_dir / "test_trading_state.db"
    yield db_file
    # Cleanup - database fixture disposes engine first
    try:
        if db_file.exists():
            db_file.unlink()
    except PermissionError:
        pass
    try:
        tmp_dir.rmdir()
    except (OSError, PermissionError):
        pass


@pytest.fixture
def database(db_path: Path) -> Generator[OperationalDatabase, None, None]:
    """Create and initialize an operational database."""
    db = OperationalDatabase(db_path)
    db.initialize()

    # Create a default trading session so FK constraints pass
    from storage.repositories import OperationalRepository
    repo = OperationalRepository(db)
    repo.create_trading_session(
        session_id="test_session",
        mode="PAPER",
        trading_date="2026-07-15",
    )

    yield db
    db.dispose()


@pytest.fixture
def portfolio_service(database: OperationalDatabase) -> PaperPortfolioService:
    """Create a portfolio service with test capital."""
    service = PaperPortfolioService(
        database,
        initial_capital=TEST_CAPITAL,
        session_id="test_session",
    )
    service.initialize()
    return service


@pytest.fixture
def accounting_repo(database: OperationalDatabase) -> PortfolioAccountingRepository:
    """Create an accounting repository."""
    return PortfolioAccountingRepository(database)


# ---------------------------------------------------------------------------
# Kill-switch persistence tests
# ---------------------------------------------------------------------------


class TestKillSwitchPersistence:
    """Kill-switch must survive restart and support authorized clearing."""

    def test_restart_with_active_kill_switch(self, database: OperationalDatabase) -> None:
        """Restart must not clear the kill switch."""
        # First instance: trigger kill switch
        ks1 = KillSwitch(database=database)
        ks1.trigger("Test trigger", source="daily_loss")
        assert ks1.active is True

        # Second instance (simulates restart): must still be active
        ks2 = KillSwitch(database=database)
        assert ks2.active is True
        state = ks2.state()
        assert state.reason == "Test trigger"
        assert state.source == "daily_loss"

    def test_authorized_clear(self, database: OperationalDatabase) -> None:
        """Authorized clear must deactivate the kill switch."""
        ks = KillSwitch(database=database)
        ks.trigger("Test trigger", source="daily_loss")
        assert ks.active is True

        # Authorized clear
        ks.reset()
        assert ks.active is False

        # Restart: must remain cleared
        ks2 = KillSwitch(database=database)
        assert ks2.active is False

    def test_unauthorized_clear_does_not_work(self, database: OperationalDatabase) -> None:
        """Direct in-memory manipulation without persistence must not affect DB."""
        ks = KillSwitch(database=database)
        ks.trigger("Test trigger", source="daily_loss")
        assert ks.active is True

        # Simulate unauthorized attempt: create new instance without resetting
        # The persisted state still shows active
        ks2 = KillSwitch(database=database)
        assert ks2.active is True

        # Even if we create an instance that doesn't load from DB...
        ks3 = KillSwitch()  # No database, in-memory only
        ks3.reset()  # Reset in-memory only
        assert ks3.active is False

        # The persisted state is still active
        ks4 = KillSwitch(database=database)
        assert ks4.active is True

    def test_concurrent_trigger(self, database: OperationalDatabase) -> None:
        """Multiple sequential triggers must record the latest."""
        ks1 = KillSwitch(database=database)
        ks1.trigger("First trigger", source="daily_loss")
        assert ks1.active is True

        # New instance loads from DB and sees active
        ks2 = KillSwitch(database=database)
        assert ks2.active is True

        # Second trigger overwrites
        ks2.trigger("Second trigger", source="excessive_drawdown")
        assert ks1.active is True  # ks1 still has old in-memory state
        assert ks2.active is True

        # Reload from DB shows latest state
        ks3 = KillSwitch(database=database)
        assert ks3.active is True


# ---------------------------------------------------------------------------
# Portfolio equity tests
# ---------------------------------------------------------------------------


class TestPortfolioEquity:
    """Current equity must be derived from persisted authoritative values."""

    def test_initial_equity(self, portfolio_service: PaperPortfolioService) -> None:
        """Initial equity must equal initial capital."""
        assert portfolio_service.get_current_equity() == TEST_CAPITAL

    def test_equity_after_profit(self, portfolio_service: PaperPortfolioService) -> None:
        """Profit must increase equity."""
        portfolio_service.record_realized_pnl(50_000.0)
        assert portfolio_service.get_current_equity() == TEST_CAPITAL + 50_000.0

    def test_equity_after_loss(self, portfolio_service: PaperPortfolioService) -> None:
        """Loss must decrease equity."""
        portfolio_service.record_realized_pnl(-30_000.0)
        assert portfolio_service.get_current_equity() == TEST_CAPITAL - 30_000.0

    def test_costs_reduce_equity(self, portfolio_service: PaperPortfolioService) -> None:
        """Brokerage, statutory costs, and slippage must reduce equity."""
        portfolio_service.record_realized_pnl(
            0.0,
            brokerage=500.0,
            statutory_costs=300.0,
            slippage=200.0,
        )
        # Costs reduce equity: -500 -300 -200 = -1000
        # Equity = initial_capital + 0 - 1000 = TEST_CAPITAL - 1000
        expected = TEST_CAPITAL - 1000.0
        assert abs(portfolio_service.get_current_equity() - expected) < 0.01

    def test_slippage_reduces_equity(self, portfolio_service: PaperPortfolioService) -> None:
        """Slippage must reduce equity independently."""
        portfolio_service.record_realized_pnl(10_000.0, slippage=1_000.0)
        # Equity = initial + 10000 - 0 - 0 - 1000 = TEST_CAPITAL + 9000
        expected = TEST_CAPITAL + 9_000.0
        assert abs(portfolio_service.get_current_equity() - expected) < 0.01

    def test_restart_preserves_equity(self, database: OperationalDatabase) -> None:
        """Restart must not reset capital."""
        from storage.repositories import OperationalRepository
        repo = OperationalRepository(database)
        repo.create_trading_session(session_id="session_1", mode="PAPER")
        repo.create_trading_session(session_id="session_2", mode="PAPER")

        # First session
        ps1 = PaperPortfolioService(
            database,
            initial_capital=TEST_CAPITAL,
            session_id="session_1",
        )
        ps1.initialize()
        ps1.record_realized_pnl(25_000.0)

        # Second session (simulates restart)
        ps2 = PaperPortfolioService(
            database,
            initial_capital=TEST_CAPITAL,
            session_id="session_2",
        )
        ps2.initialize()
        assert abs(ps2.get_current_equity() - (TEST_CAPITAL + 25_000.0)) < 0.01

    def test_pnl_realized_once(self, portfolio_service: PaperPortfolioService) -> None:
        """P&L must be realized exactly once per close."""
        portfolio_service.record_realized_pnl(10_000.0)
        portfolio_service.record_realized_pnl(10_000.0)
        # Should be 20k total, not 10k + compounding
        expected = TEST_CAPITAL + 20_000.0
        assert abs(portfolio_service.get_current_equity() - expected) < 0.01

    def test_reconciliation_identity(self, accounting_repo: PortfolioAccountingRepository) -> None:
        """Equity identity must hold within tolerance."""
        state = accounting_repo.load_portfolio_state()
        state["initial_capital"] = TEST_CAPITAL
        state["realized_pnl"] = 50_000.0
        state["unrealized_pnl"] = 10_000.0
        state["accrued_brokerage"] = 1_000.0
        state["accrued_statutory_costs"] = 500.0
        state["accrued_slippage"] = 300.0
        state["margin_used"] = 200_000.0
        state["margin_reserved"] = 50_000.0

        saved = accounting_repo.save_portfolio_state(
            session_id="test_session",
            state=state,
        )

        issues = saved.get("_reconciliation_issues", [])
        assert len(issues) == 0, f"Reconciliation issues: {issues}"


# ---------------------------------------------------------------------------
# Risk and margin tests
# ---------------------------------------------------------------------------


class TestRiskAndMargin:
    """Risk and margin reservation tests."""

    def test_pending_order_risk(self, portfolio_service: PaperPortfolioService) -> None:
        """Pending orders must reserve risk and margin."""
        portfolio_service.reserve_pending_order_risk(risk=10_000.0, margin=50_000.0)
        state = portfolio_service.get_state()
        assert state["pending_order_risk"] == 10_000.0
        assert state["margin_reserved"] == 50_000.0

    def test_partial_fill_risk(self, portfolio_service: PaperPortfolioService) -> None:
        """Partial fills must reserve actual remaining exposure."""
        portfolio_service.reserve_partial_fill_risk(risk=5_000.0, margin=20_000.0)
        state = portfolio_service.get_state()
        assert state["partial_fill_risk"] == 5_000.0
        assert state["margin_reserved"] == 20_000.0

    def test_margin_reservation(self, portfolio_service: PaperPortfolioService) -> None:
        """Margin used must be tracked."""
        portfolio_service.open_strategy(risk_amount=15_000.0, margin_required=100_000.0)
        state = portfolio_service.get_state()
        assert state["capital_at_risk"] == 15_000.0
        assert state["margin_used"] == 100_000.0

        portfolio_service.close_strategy(risk_amount=15_000.0, margin_required=100_000.0)
        state = portfolio_service.get_state()
        assert state["capital_at_risk"] == 0.0
        assert state["margin_used"] == 0.0

    def test_peak_equity(self, portfolio_service: PaperPortfolioService) -> None:
        """Peak equity must track the highest equity reached."""
        assert portfolio_service.get_peak_equity() == TEST_CAPITAL
        portfolio_service.record_realized_pnl(20_000.0)
        assert portfolio_service.get_peak_equity() == TEST_CAPITAL + 20_000.0
        # Loss should not reduce peak
        portfolio_service.record_realized_pnl(-10_000.0)
        assert portfolio_service.get_peak_equity() == TEST_CAPITAL + 20_000.0

    def test_drawdown(self, portfolio_service: PaperPortfolioService) -> None:
        """Drawdown must be calculated correctly."""
        portfolio_service.record_realized_pnl(50_000.0)
        peak = portfolio_service.get_peak_equity()
        assert peak == TEST_CAPITAL + 50_000.0

        portfolio_service.record_realized_pnl(-30_000.0)
        current = portfolio_service.get_current_equity()
        expected_dd = (peak - current) / peak * 100.0
        assert abs(portfolio_service.get_drawdown() - expected_dd) < 0.01


# ---------------------------------------------------------------------------
# Daily counter tests
# ---------------------------------------------------------------------------


class TestDailyCounters:
    """Daily counter management tests."""

    def test_daily_counter_reset(self, portfolio_service: PaperPortfolioService) -> None:
        """New trading dates may reset daily counters only."""
        portfolio_service.record_realized_pnl(10_000.0)
        portfolio_service.record_realized_pnl(-5_000.0)
        state = portfolio_service.get_state()
        assert state["daily_realized_pnl"] == 5_000.0
        assert state["trade_count"] == 2
        assert state["winning_trades"] == 1
        assert state["losing_trades"] == 1

        # Reset daily counters
        portfolio_service.reset_daily_counters()
        state = portfolio_service.get_state()
        assert state["daily_realized_pnl"] == 0.0
        assert state["trade_count"] == 0
        assert state["winning_trades"] == 0
        assert state["losing_trades"] == 0

    def test_unresolved_state_survives_daily_reset(
        self, portfolio_service: PaperPortfolioService
    ) -> None:
        """New dates must not clear unresolved orders, positions, or reconciliation blocks."""
        portfolio_service.reserve_pending_order_risk(risk=10_000.0, margin=50_000.0)
        portfolio_service.record_realized_pnl(5_000.0)

        # Reset daily counters
        portfolio_service.reset_daily_counters()

        # Pending order risk must survive
        state = portfolio_service.get_state()
        assert state["pending_order_risk"] == 10_000.0
        assert state["margin_reserved"] == 50_000.0
        # Realized P&L must survive (cumulative, not daily)
        assert state["realized_pnl"] == 5_000.0


# ---------------------------------------------------------------------------
# Equity guard tests
# ---------------------------------------------------------------------------


class TestEquityGuard:
    """Zero or negative equity must block new entries."""

    def test_zero_equity(self, database: OperationalDatabase) -> None:
        """Zero equity must block new entries."""
        ps = PaperPortfolioService(
            database,
            initial_capital=0.0,
            session_id="test_session",
        )
        ps.initialize()
        can_trade, reason = ps.can_trade()
        assert can_trade is False
        assert "zero" in reason.lower()

    def test_negative_equity(self, database: OperationalDatabase) -> None:
        """Negative equity must block new entries."""
        ps = PaperPortfolioService(
            database,
            initial_capital=TEST_CAPITAL,
            session_id="test_session",
        )
        ps.initialize()
        # Record a large loss to make equity negative
        ps.record_realized_pnl(-TEST_CAPITAL * 2)
        can_trade, reason = ps.can_trade()
        assert can_trade is False
        assert "negative" in reason.lower()

    def test_positive_equity_allows_trading(
        self, portfolio_service: PaperPortfolioService
    ) -> None:
        """Positive equity must allow trading."""
        can_trade, reason = portfolio_service.can_trade()
        assert can_trade is True
        assert "allowed" in reason.lower()

    def test_kill_switch_blocks_trading(
        self, portfolio_service: PaperPortfolioService
    ) -> None:
        """Active kill switch must block trading."""
        portfolio_service.trigger_kill_switch(
            "Test block", source="daily_loss"
        )
        can_trade, reason = portfolio_service.can_trade()
        assert can_trade is False
        assert "kill switch" in reason.lower()


# ---------------------------------------------------------------------------
# Optimistic locking tests
# ---------------------------------------------------------------------------


class TestOptimisticLocking:
    """Concurrent updates must use optimistic locking."""

    def test_optimistic_lock_conflict(self, database: OperationalDatabase) -> None:
        """Optimistic-lock conflict must raise an error."""
        from storage.repositories import OperationalRepository
        repo = OperationalRepository(database)
        repo.create_trading_session(session_id="session_1", mode="PAPER")
        repo.create_trading_session(session_id="session_2", mode="PAPER")

        ps1 = PaperPortfolioService(
            database,
            initial_capital=TEST_CAPITAL,
            session_id="session_1",
        )
        ps1.initialize()
        ps1.record_realized_pnl(10_000.0)

        # ps2 loads the same state
        ps2 = PaperPortfolioService(
            database,
            initial_capital=TEST_CAPITAL,
            session_id="session_2",
        )
        ps2.initialize()

        # Both have version 1. ps1 writes (version goes to 2).
        ps1.record_realized_pnl(5_000.0)

        # ps2 tries to write with expected_version=1, which is now stale
        with pytest.raises(OptimisticLockError):
            ps2.persist_with_optimistic_lock(expected_version=1)


# ---------------------------------------------------------------------------
# Trade tracking tests
# ---------------------------------------------------------------------------


class TestTradeTracking:
    """Trade counting and win/loss tracking."""

    def test_trade_count(self, portfolio_service: PaperPortfolioService) -> None:
        """Trade count must increment on realized P&L."""
        portfolio_service.record_realized_pnl(10_000.0)
        portfolio_service.record_realized_pnl(-5_000.0)
        portfolio_service.record_realized_pnl(2_000.0)
        state = portfolio_service.get_state()
        assert state["trade_count"] == 3

    def test_win_loss_tracking(self, portfolio_service: PaperPortfolioService) -> None:
        """Winning and losing trades must be tracked correctly."""
        portfolio_service.record_realized_pnl(10_000.0)
        portfolio_service.record_realized_pnl(5_000.0)
        portfolio_service.record_realized_pnl(-3_000.0)
        state = portfolio_service.get_state()
        assert state["winning_trades"] == 2
        assert state["losing_trades"] == 1

    def test_consecutive_losses(self, portfolio_service: PaperPortfolioService) -> None:
        """Consecutive losses must be tracked correctly."""
        portfolio_service.record_realized_pnl(-1_000.0)
        portfolio_service.record_realized_pnl(-2_000.0)
        portfolio_service.record_realized_pnl(-3_000.0)
        state = portfolio_service.get_state()
        assert state["consecutive_losses"] == 3

        # Win resets consecutive losses
        portfolio_service.record_realized_pnl(10_000.0)
        state = portfolio_service.get_state()
        assert state["consecutive_losses"] == 0

    def test_open_strategy_count(self, portfolio_service: PaperPortfolioService) -> None:
        """Open strategy count must track strategies."""
        portfolio_service.open_strategy(risk_amount=10_000.0, margin_required=50_000.0)
        state = portfolio_service.get_state()
        assert state["open_strategy_count"] == 1

        portfolio_service.open_strategy(risk_amount=15_000.0, margin_required=75_000.0)
        state = portfolio_service.get_state()
        assert state["open_strategy_count"] == 2

        # Close one
        portfolio_service.record_realized_pnl(5_000.0)
        state = portfolio_service.get_state()
        assert state["open_strategy_count"] == 1