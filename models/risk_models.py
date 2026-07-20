"""Risk decision models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PortfolioAccounting:
    """Authoritative point-in-time equity and reserved-exposure snapshot."""

    initial_capital: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    accrued_costs: float = 0.0
    pending_order_risk: float = 0.0
    partial_fill_risk: float = 0.0
    pending_order_margin: float = 0.0
    partial_fill_margin: float = 0.0

    @property
    def current_equity(self) -> float:
        """Return initial plus P&L less accrued costs."""

        return self.initial_capital + self.realized_pnl + self.unrealized_pnl - self.accrued_costs

    @property
    def reserved_risk(self) -> float:
        return max(self.pending_order_risk, 0.0) + max(self.partial_fill_risk, 0.0)

    @property
    def reserved_margin(self) -> float:
        return max(self.pending_order_margin, 0.0) + max(self.partial_fill_margin, 0.0)


@dataclass(frozen=True)
class ClearAuthorization:
    """Explicit authorization for clearing the kill switch."""

    operator: str
    reason: str
    authorized_at: datetime | None = None


@dataclass(frozen=True)
class KillSwitchState:
    """Authoritative kill-switch state."""

    active: bool
    reason: str | None
    triggered_at: datetime | None
    source: str | None = None
    cleared_at: datetime | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass(frozen=True)
class RiskDecision:
    """Risk engine decision for a strategy plan or order."""

    approved: bool
    timestamp: datetime
    reason: str
    max_allowed_quantity: int | None = None
    capital_at_risk: float = 0.0
    margin_required: float = 0.0
    checks: dict[str, bool] = field(default_factory=dict)
    rejection_reason: str | None = None
    allowed_quantity: int | None = None
    max_risk_amount: float = 0.0
    risk_used: float = 0.0
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
