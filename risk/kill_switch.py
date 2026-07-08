"""System-wide risk kill switch."""

from __future__ import annotations

from dataclasses import dataclass

from utils.time_utils import now_ist


@dataclass(frozen=True)
class KillSwitchState:
    """Current kill switch state."""

    active: bool
    reason: str | None
    triggered_at: str | None


class KillSwitch:
    """In-memory kill switch for blocking new risk approvals."""

    def __init__(self) -> None:
        """Create an inactive kill switch."""

        self._active = False
        self._reason: str | None = None
        self._triggered_at: str | None = None

    @property
    def active(self) -> bool:
        """Return true if the kill switch is active."""

        return self._active

    def trigger(self, reason: str) -> KillSwitchState:
        """Activate the kill switch."""

        self._active = True
        self._reason = reason
        self._triggered_at = now_ist().isoformat()
        return self.state()

    def reset(self) -> KillSwitchState:
        """Reset the kill switch."""

        self._active = False
        self._reason = None
        self._triggered_at = None
        return self.state()

    def state(self) -> KillSwitchState:
        """Return current state."""

        return KillSwitchState(self._active, self._reason, self._triggered_at)
