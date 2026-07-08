"""Reporting models for trades and system health."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class TradeReport:
    """Daily or session trade performance summary."""

    report_date: date
    generated_at: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    gross_pnl: float
    net_pnl: float
    brokerage: float = 0.0
    max_drawdown: float = 0.0
    strategy_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LatencyReport:
    """Latency summary for market data, decisions, and execution."""

    report_date: date
    generated_at: datetime
    component: str
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
