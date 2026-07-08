"""Risk decision models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
