"""Prediction and regime model outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models.common import RegimeType


@dataclass(frozen=True)
class RegimeResult:
    """Detected market regime with confidence and supporting features."""

    symbol: str
    timestamp: datetime
    regime: RegimeType
    confidence: float
    lookback_bars: int
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    regime_name: str | None = None
    trend_score: float = 0.0
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    event_risk_score: float = 0.0
    confidence_score: float = 0.0
    allowed_to_trade: bool = False
    preferred_strategy_types: tuple[str, ...] = field(default_factory=tuple)
    forbidden_strategy_types: tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""


@dataclass(frozen=True)
class PredictionResult:
    """Model prediction consumed by strategy selection."""

    symbol: str
    timestamp: datetime
    predicted_direction: str
    confidence: float
    horizon_minutes: int
    regime: RegimeType
    expected_return_percent: float | None = None
    expected_volatility_percent: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    underlying: str | None = None
    bullish_probability: float = 0.0
    bearish_probability: float = 0.0
    sideways_probability: float = 0.0
    expected_move_percent: float = 0.0
    expected_range_low: float | None = None
    expected_range_high: float | None = None
    volatility_state: str = "unknown"
    breakout_failure_probability: float = 0.0
    recommended_strategy_type: str | None = None
    confidence_score: float = 0.0
    trade_allowed: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
