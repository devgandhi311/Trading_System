"""SQLAlchemy models for authoritative operational state."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from utils.time_utils import now_ist


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    __tablename__ = "schema_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_ist, nullable=False)


class TradingSession(Base):
    """Authoritative trading session with full lifecycle tracking."""
    __tablename__ = "trading_sessions"
    __table_args__ = (
        UniqueConstraint("trading_date", "mode", name="uq_session_date_mode"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    trading_date: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="OPEN", nullable=False)
    previous_state: Mapped[str | None] = mapped_column(String(32))
    state_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str | None] = mapped_column(Text)
    recovery_reason: Mapped[str | None] = mapped_column(Text)
    final_equity_reference: Mapped[float | None] = mapped_column(Float)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class DecisionCycle(Base):
    __tablename__ = "decision_cycles"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class DecisionContextRecord(Base):
    __tablename__ = "decision_contexts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    decision_cycle_id: Mapped[str] = mapped_column(ForeignKey("decision_cycles.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="CASCADE"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    market_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    feature_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    regime: Mapped[str | None] = mapped_column(String(64))
    regime_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    regime_warnings: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    prediction_probabilities: Mapped[tuple[float, float, float] | None] = mapped_column(JSON)
    prediction_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    expected_return_percent: Mapped[float | None] = mapped_column(Float)
    expected_range_low: Mapped[float | None] = mapped_column(Float)
    expected_range_high: Mapped[float | None] = mapped_column(Float)
    model_names: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    model_versions: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    volatility_forecast: Mapped[float | None] = mapped_column(Float)
    volatility_model_requested: Mapped[str | None] = mapped_column(String(64))
    volatility_model_used: Mapped[str | None] = mapped_column(String(64))
    convergence_status: Mapped[str | None] = mapped_column(String(64))
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_reason: Mapped[str | None] = mapped_column(Text)
    expected_expiry_price: Mapped[float | None] = mapped_column(Float)
    expected_move: Mapped[float | None] = mapped_column(Float)
    range_68: Mapped[tuple[float, float] | None] = mapped_column(JSON)
    range_95: Mapped[tuple[float, float] | None] = mapped_column(JSON)
    tail_risk_score: Mapped[float | None] = mapped_column(Float)
    news_sentiment: Mapped[float | None] = mapped_column(Float)
    event_risk_level: Mapped[str | None] = mapped_column(String(32))
    liquidity_score: Mapped[float | None] = mapped_column(Float)
    entry_context_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    invalid_reason: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class OrderIntentRecord(Base):
    __tablename__ = "order_intents"
    __table_args__ = (
        UniqueConstraint("client_order_id"),
        UniqueConstraint("idempotency_key"),
        UniqueConstraint("broker_order_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="RESTRICT"), nullable=False)
    basket_id: Mapped[str | None] = mapped_column(String(128))
    parent_intent_id: Mapped[int | None] = mapped_column(ForeignKey("order_intents.id", ondelete="RESTRICT"))
    strategy_id: Mapped[str | None] = mapped_column(String(128))
    leg_id: Mapped[str | None] = mapped_column(String(128))
    instrument_token: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    limit_price: Mapped[float | None] = mapped_column(Float)
    trigger_price: Mapped[float | None] = mapped_column(Float)
    product_type: Mapped[str] = mapped_column(String(32), nullable=False)
    validity: Mapped[str] = mapped_column(String(16), nullable=False)
    local_state: Mapped[str] = mapped_column(String(32), nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(128))
    attempt_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reconciliation_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    failure_classification: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class OrderEvent(Base):
    __tablename__ = "order_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_intent_id: Mapped[int] = mapped_column(ForeignKey("order_intents.id", ondelete="CASCADE"), nullable=False)
    from_state: Mapped[str | None] = mapped_column(String(32))
    to_state: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class FillRecord(Base):
    __tablename__ = "fills"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    order_intent_id: Mapped[int] = mapped_column(ForeignKey("order_intents.id", ondelete="RESTRICT"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    transaction_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class PositionRecord(Base):
    __tablename__ = "positions"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="RESTRICT"), nullable=False)
    strategy_id: Mapped[str | None] = mapped_column(String(128))
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    average_price: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="RESTRICT"), nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, nullable=False)
    current_equity: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    accrued_costs: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class RiskDecisionRecord(Base):
    __tablename__ = "risk_decisions"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="RESTRICT"), nullable=False)
    decision_cycle_id: Mapped[str | None] = mapped_column(ForeignKey("decision_cycles.id", ondelete="SET NULL"))
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class ReconciliationEvent(Base):
    __tablename__ = "reconciliation_events"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="RESTRICT"), nullable=False)
    outcome: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_blocked: Mapped[bool] = mapped_column(Boolean, nullable=False)
    exits_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    affected_order_ids: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    affected_position_ids: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    local_state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    broker_state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    mismatch_details: Mapped[tuple[dict[str, Any], ...]] = mapped_column(JSON, default=tuple, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    matched_orders: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    matched_fills: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    matched_positions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    repaired_items: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    blocking_reasons: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    warnings: Mapped[tuple[str, ...]] = mapped_column(JSON, default=tuple, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class KillSwitchEvent(Base):
    __tablename__ = "kill_switch_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("trading_sessions.id", ondelete="SET NULL"))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class MarketSnapshotRecord(Base):
    __tablename__ = "market_snapshots"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="RESTRICT"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    instrument_token: Mapped[str] = mapped_column(String(128), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class ExitDecisionRecord(Base):
    __tablename__ = "exit_decisions"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="RESTRICT"), nullable=False)
    position_id: Mapped[str] = mapped_column(ForeignKey("positions.id", ondelete="RESTRICT"), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class SessionTransitionRecord(Base):
    """Persistent log of every session state transition."""
    __tablename__ = "session_transitions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("trading_sessions.id", ondelete="CASCADE"), nullable=False)
    from_state: Mapped[str | None] = mapped_column(String(32))
    to_state: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)