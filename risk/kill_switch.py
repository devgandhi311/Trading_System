"""System-wide risk kill switch with persistence support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.risk_models import ClearAuthorization, KillSwitchState
from storage.database import OperationalDatabase
from storage.repositories import PortfolioAccountingRepository
from utils.time_utils import now_ist


class KillSwitch:
    """Kill switch that can operate in-memory or backed by persistent storage.

    When a PortfolioAccountingRepository is provided, all state changes are
    persisted. Restart must not clear the kill switch.
    """

    def __init__(
        self,
        database: OperationalDatabase | None = None,
    ) -> None:
        """Create an inactive kill switch, optionally with persistence."""
        self._active = False
        self._reason: str | None = None
        self._triggered_at: str | None = None
        self._source: str | None = None
        self._repo: PortfolioAccountingRepository | None = None
        if database is not None:
            self._repo = PortfolioAccountingRepository(database)
            self._load_from_db()

    def _load_from_db(self) -> None:
        """Load persisted kill-switch state on construction.

        Restart must not clear the kill switch.
        """
        if self._repo is None:
            return
        state = self._repo.current_kill_switch()
        if state is not None:
            self._active = bool(state.active)
            self._reason = state.reason
            self._triggered_at = state.triggered_at.isoformat() if state.triggered_at else None
            self._source = state.source

    @property
    def active(self) -> bool:
        """Return true if the kill switch is active."""
        return self._active

    def trigger(
        self,
        reason: str,
        *,
        source: str = "manual_operator_action",
        metadata: Mapping[str, Any] | None = None,
    ) -> KillSwitchState:
        """Activate the kill switch.

        Supports persisted triggers from kill-switch sources.
        """
        if self._repo is not None:
            state = self._repo.trigger_kill_switch(
                reason=reason,
                source=source,
                metadata=metadata,
            )
            self._active = state.active
            self._reason = state.reason
            self._triggered_at = state.triggered_at.isoformat() if state.triggered_at else None
            self._source = state.source
            return state
        self._active = True
        self._reason = reason
        self._triggered_at = now_ist().isoformat()
        self._source = source
        return KillSwitchState(
            active=True,
            reason=reason,
            triggered_at=now_ist(),
            source=source,
        )

    def reset(self, authorization: ClearAuthorization | None = None) -> KillSwitchState:
        """Reset the kill switch.

        When persistence is active, this is the explicit authorized method.
        """
        auth = authorization or ClearAuthorization(operator="authorized_operator", reason="Authorized operator reset")
        if self._repo is not None:
            state = self._repo.clear_kill_switch(
                authorization=auth,
            )
            self._active = state.active
            self._reason = state.reason
            self._triggered_at = state.triggered_at.isoformat() if state.triggered_at else None
            self._source = state.source
            return state
        self._active = False
        self._reason = None
        self._triggered_at = None
        self._source = None
        return KillSwitchState(
            active=False,
            reason=None,
            triggered_at=None,
            source=None,
            cleared_at=now_ist(),
        )

    def state(self) -> KillSwitchState:
        """Return current state."""
        return KillSwitchState(
            active=self._active,
            reason=self._reason,
            triggered_at=self._triggered_at,
            source=self._source,
        )