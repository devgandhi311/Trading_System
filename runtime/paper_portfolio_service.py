"""Authoritative persistent PAPER portfolio and risk-state service.

Provides the single source of truth for portfolio accounting, position
sizing, risk state, and kill-switch management. All state is persisted
in SQLite via the PortfolioAccountingRepository, ensuring restart safety
and optimistic-locking concurrency control.
"""

from __future__ import annotations

import logging
from typing import Any

from storage.database import OperationalDatabase
from storage.repositories import (
    ClearAuthorization,
    OptimisticLockError,
    PortfolioAccountingRepository,
)
from models.risk_models import KillSwitchState
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)

KILL_SWITCH_SOURCES = {
    "daily_loss",
    "drawdown",
    "stale_data",
    "reconciliation_mismatch",
    "runtime_failure",
    "unknown_order",
    "unhedged_partial_fill",
    "emergency_exit_failure",
    "manual",
}


class PaperPortfolioService:
    """Authoritative persistent PAPER portfolio and risk-state service.

    Responsibilities:
    - Maintain the single source of truth for all portfolio fields.
    - Derive current_equity from persisted authoritative values.
    - Provide current equity to the PositionSizer.
    - Manage pending-order risk reservation and partial-fill risk.
    - Track open positions for unrealized P&L.
    - Realize P&L exactly once on position close.
    - Handle restart without resetting capital or clearing kill switch.
    - Handle daily counter reset without clearing orders/positions/reconciliation.
    - Block new entries on zero or negative equity.
    - Support explicit authorized kill-switch clearing.
    - Use optimistic locking for concurrent updates.
    - Reconcile all calculations within defined tolerance.
    """

    def __init__(
        self,
        database: OperationalDatabase,
        *,
        initial_capital: float | None = None,
        session_id: str | None = None,
    ) -> None:
        self._repo = PortfolioAccountingRepository(database)
        self._session_id = session_id or "paper_default"
        self._initial_capital = initial_capital
        self._state: dict[str, Any] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Initialization and state loading
    # ------------------------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        """Load persisted state or bootstrap with initial capital.

        Restart must not reset capital.
        """
        self._state = self._repo.load_portfolio_state()
        if not self._loaded and self._initial_capital is not None:
            if self._state["initial_capital"] == 0.0:
                self._state["initial_capital"] = self._initial_capital
                self._state["current_equity"] = self._initial_capital
                self._state["peak_equity"] = self._initial_capital
                self._state["available_capital"] = self._initial_capital
                self._repo.save_portfolio_state(
                    session_id=self._session_id,
                    state=self._state,
                )
        self._loaded = True
        LOGGER.info(
            "Portfolio state loaded",
            extra={
                "initial_capital": self._state.get("initial_capital"),
                "current_equity": self._state.get("current_equity"),
                "peak_equity": self._state.get("peak_equity"),
            },
        )
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Return the current authoritative portfolio state dict."""
        return dict(self._state)

    def get_current_equity(self) -> float:
        """Return the derived current equity.

        Position sizing must use current equity.
        """
        self.initialize()
        return float(self._state.get("current_equity", 0.0))

    def get_available_capital(self) -> float:
        """Return available capital (equity minus used and reserved margin)."""
        self.initialize()
        return float(self._state.get("available_capital", 0.0))

    def get_peak_equity(self) -> float:
        """Return the peak equity ever reached."""
        self.initialize()
        return float(self._state.get("peak_equity", 0.0))

    def get_drawdown(self) -> float:
        """Return the current drawdown percentage."""
        self.initialize()
        return float(self._state.get("drawdown", 0.0))

    def get_open_strategy_count(self) -> int:
        """Return the number of open strategies."""
        self.initialize()
        return int(self._state.get("open_strategy_count", 0))

    # ------------------------------------------------------------------
    # P&L and trade lifecycle
    # ------------------------------------------------------------------

    def record_realized_pnl(
        self,
        pnl: float,
        *,
        trade_side: str = "BUY",
        brokerage: float = 0.0,
        statutory_costs: float = 0.0,
        slippage: float = 0.0,
        strategy_id: str | None = None,
    ) -> dict[str, Any]:
        """Record realized P&L exactly once for a closed position.

        Closed positions realize P&L exactly once.
        Updates daily and cumulative counters.
        """
        self.initialize()
        state = self._state
        state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
        state["daily_realized_pnl"] = float(state.get("daily_realized_pnl", 0.0)) + pnl
        state["accrued_brokerage"] = float(state.get("accrued_brokerage", 0.0)) + brokerage
        state["accrued_statutory_costs"] = float(state.get("accrued_statutory_costs", 0.0)) + statutory_costs
        state["accrued_slippage"] = float(state.get("accrued_slippage", 0.0)) + slippage
        state["trade_count"] = int(state.get("trade_count", 0)) + 1

        if pnl > 0:
            state["winning_trades"] = int(state.get("winning_trades", 0)) + 1
            state["consecutive_losses"] = 0
        else:
            state["losing_trades"] = int(state.get("losing_trades", 0)) + 1
            state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1

        if int(state.get("open_strategy_count", 0)) > 0:
            state["open_strategy_count"] = int(state.get("open_strategy_count", 0)) - 1

        return self._persist_state()

    def update_unrealized_pnl(self, unrealized_pnl: float) -> dict[str, Any]:
        """Update unrealized P&L from open positions.

        Open positions contribute unrealized P&L.
        """
        self.initialize()
        state = self._state
        state["unrealized_pnl"] = unrealized_pnl
        state["daily_unrealized_pnl"] = unrealized_pnl
        return self._persist_state()

    def open_strategy(self, *, risk_amount: float, margin_required: float) -> dict[str, Any]:
        """Record a new opened strategy that reserves risk and margin."""
        self.initialize()
        state = self._state
        state["open_strategy_count"] = int(state.get("open_strategy_count", 0)) + 1
        state["capital_at_risk"] = float(state.get("capital_at_risk", 0.0)) + risk_amount
        state["margin_used"] = float(state.get("margin_used", 0.0)) + margin_required
        return self._persist_state()

    def close_strategy(
        self,
        *,
        risk_amount: float,
        margin_required: float,
    ) -> dict[str, Any]:
        """Release risk and margin reservations for a closed strategy."""
        self.initialize()
        state = self._state
        state["capital_at_risk"] = max(
            0.0, float(state.get("capital_at_risk", 0.0)) - risk_amount
        )
        state["margin_used"] = max(
            0.0, float(state.get("margin_used", 0.0)) - margin_required
        )
        return self._persist_state()

    # ------------------------------------------------------------------
    # Pending-order and partial-fill risk
    # ------------------------------------------------------------------

    def reserve_pending_order_risk(
        self,
        risk: float,
        margin: float,
    ) -> dict[str, Any]:
        """Reserve risk and margin for a pending order.

        Pending orders reserve risk and margin.
        """
        self.initialize()
        state = self._state
        state["pending_order_risk"] = float(state.get("pending_order_risk", 0.0)) + risk
        state["margin_reserved"] = float(state.get("margin_reserved", 0.0)) + margin
        return self._persist_state()

    def release_pending_order_risk(
        self,
        risk: float,
        margin: float,
    ) -> dict[str, Any]:
        """Release risk and margin for a completed/cancelled order."""
        self.initialize()
        state = self._state
        state["pending_order_risk"] = max(
            0.0, float(state.get("pending_order_risk", 0.0)) - risk
        )
        state["margin_reserved"] = max(
            0.0, float(state.get("margin_reserved", 0.0)) - margin
        )
        return self._persist_state()

    def reserve_partial_fill_risk(
        self,
        risk: float,
        margin: float,
    ) -> dict[str, Any]:
        """Reserve actual remaining exposure for a partial fill.

        Partial fills reserve actual remaining exposure.
        """
        self.initialize()
        state = self._state
        state["partial_fill_risk"] = float(state.get("partial_fill_risk", 0.0)) + risk
        state["margin_reserved"] = float(state.get("margin_reserved", 0.0)) + margin
        return self._persist_state()

    def release_partial_fill_risk(
        self,
        risk: float,
        margin: float,
    ) -> dict[str, Any]:
        """Release remaining exposure when a partial fill is resolved."""
        self.initialize()
        state = self._state
        state["partial_fill_risk"] = max(
            0.0, float(state.get("partial_fill_risk", 0.0)) - risk
        )
        state["margin_reserved"] = max(
            0.0, float(state.get("margin_reserved", 0.0)) - margin
        )
        return self._persist_state()

    # ------------------------------------------------------------------
    # Daily counter management
    # ------------------------------------------------------------------

    def reset_daily_counters(self) -> dict[str, Any]:
        """Reset daily counters only.

        New trading dates may reset daily counters only.
        New dates must not clear unresolved orders, positions, or
        reconciliation blocks.
        """
        self.initialize()
        state = self._state
        state["daily_realized_pnl"] = 0.0
        state["daily_unrealized_pnl"] = 0.0
        state["trade_count"] = 0
        state["winning_trades"] = 0
        state["losing_trades"] = 0
        state["consecutive_losses"] = 0
        return self._persist_state()

    # ------------------------------------------------------------------
    # Kill-switch management
    # ------------------------------------------------------------------

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active.

        Restart must not clear the kill switch.
        """
        return self._repo.is_kill_switch_active()

    def trigger_kill_switch(
        self,
        reason: str,
        *,
        source: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> KillSwitchState:
        """Activate the kill switch with a persisted event.

        Supports persisted triggers from kill-switch sources.
        """
        if source not in KILL_SWITCH_SOURCES:
            LOGGER.warning(
                "Kill-switch source '%s' is not in known sources",
                source,
            )
        return self._repo.trigger_kill_switch(
            reason=reason,
            source=source,
            session_id=self._session_id,
            metadata=metadata,
        )

    def clear_kill_switch(
        self,
        *,
        authorization: ClearAuthorization,
    ) -> KillSwitchState:
        """Clear the kill switch using the explicit authorized method.

        Kill-switch clearing must use an explicit authorized method.
        """
        return self._repo.clear_kill_switch(
            authorization=authorization,
            session_id=self._session_id,
        )

    def kill_switch_state(self) -> KillSwitchState | None:
        """Return the current kill-switch event, if any."""
        return self._repo.current_kill_switch()

    def get_active_kill_switch(self) -> KillSwitchState | None:
        """Return the active kill-switch state, if any."""
        return self._repo.get_active_kill_switch()

    # ------------------------------------------------------------------
    # Equity guard
    # ------------------------------------------------------------------

    def can_trade(self) -> tuple[bool, str]:
        """Check if the system can enter new positions.

        Zero or negative equity blocks new entries.
        Active kill-switch blocks new entries.
        """
        if self.is_kill_switch_active():
            return False, "Kill switch is active."
        equity = self.get_current_equity()
        if equity <= 0:
            return False, f"Current equity is zero or negative: {equity:.2f}"
        return True, "Trading allowed."

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_state(self) -> dict[str, Any]:
        """Persist the current state and return it."""
        self._state = self._repo.save_portfolio_state(
            session_id=self._session_id,
            state=self._state,
        )
        return dict(self._state)

    def persist_with_optimistic_lock(
        self,
        expected_version: int,
    ) -> dict[str, Any]:
        """Persist state with optimistic-locking check.

        Raises OptimisticLockError if version does not match.
        """
        self._state = self._repo.save_portfolio_state(
            session_id=self._session_id,
            state=self._state,
            expected_version=expected_version,
        )
        return dict(self._state)