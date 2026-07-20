"""Shared runtime state models for PAPER trading only."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from enum import Enum, StrEnum
from types import UnionType
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

from models.common import Mode, OrderSide, OrderStatus, OrderType, StrategyType
from models.market_models import OptionChainSnapshot, PaperMarketSnapshot
from models.prediction_models import PredictionResult, RegimeResult
from models.strategy_models import StrategyLeg
from utils.time_utils import ensure_ist

T = TypeVar("T", bound="JsonSerializable")


class JsonSerializable:
    """Provide JSON-safe dictionary serialization for nested runtime models."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this model, enums, timestamps, and nested dataclasses."""

        return {item.name: _json_value(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_dict(cls: type[T], payload: Mapping[str, Any]) -> T:
        """Restore a model from its JSON-decoded mapping."""

        if not isinstance(payload, Mapping):
            raise ValueError(f"{cls.__name__} payload must be a mapping")
        hints = get_type_hints(cls)
        values = {item.name: _restore_value(hints[item.name], payload[item.name]) for item in fields(cls) if item.name in payload}
        return cls(**values)


class SessionState(StrEnum):
    """Durable session lifecycle states for the PAPER session scheduler.

    NOT_STARTED     — No session exists for this trading date yet.
    PREOPEN         — Before market open; pre-market checks run here.
    OPEN            — Market is open but entry may not yet be allowed.
    ENTRY_ALLOWED   — New entries, exits, and monitoring are permitted.
    EXIT_ONLY       — Entry cutoff reached; only exits and monitoring allowed.
    SQUAREOFF       — Force-squareoff window; positions being closed.
    CLOSED          — Session complete; final equity persisted.
    HOLIDAY         — Configured NSE holiday; no trading.
    FAILED          — Session ended due to unrecoverable error.
    RECOVERY_REQUIRED — Prior session requires resolution before new session.
    """

    NOT_STARTED = "NOT_STARTED"
    PREOPEN = "PREOPEN"
    OPEN = "OPEN"
    ENTRY_ALLOWED = "ENTRY_ALLOWED"
    EXIT_ONLY = "EXIT_ONLY"
    SQUAREOFF = "SQUAREOFF"
    CLOSED = "CLOSED"
    HOLIDAY = "HOLIDAY"
    FAILED = "FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


# Session phases that allow exits
EXIT_ALLOWING_PHASES = {SessionState.OPEN, SessionState.ENTRY_ALLOWED, SessionState.EXIT_ONLY, SessionState.SQUAREOFF}

# Session phases that allow new entries
ENTRY_ALLOWING_PHASES = {SessionState.ENTRY_ALLOWED}


@dataclass(frozen=True)
class PaperFill(JsonSerializable):
    """One simulated PAPER order fill."""

    fill_id: str
    order_id: str
    timestamp: datetime
    symbol: str
    instrument_token: str
    side: OrderSide
    quantity: int
    lot_size: int
    fill_price: float
    transaction_costs: float = 0.0
    slippage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(fill_id=self.fill_id, order_id=self.order_id, symbol=self.symbol, instrument_token=self.instrument_token)
        _normalize_timestamp(self, "timestamp")
        _require_positive(quantity=self.quantity, lot_size=self.lot_size, fill_price=self.fill_price)
        _require_non_negative(transaction_costs=self.transaction_costs, slippage=self.slippage)


@dataclass(frozen=True)
class PaperOrder(JsonSerializable):
    """Persistent state for one simulated PAPER order."""

    order_id: str
    created_at: datetime
    updated_at: datetime
    symbol: str
    instrument_token: str
    strategy_id: str
    leg_id: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    lot_size: int
    status: OrderStatus = OrderStatus.PENDING
    requested_price: float | None = None
    trigger_price: float | None = None
    fill_price: float | None = None
    filled_quantity: int = 0
    fills: tuple[PaperFill, ...] = ()
    rejection_reason: str | None = None
    remaining_quantity: int | None = None
    average_fill_price: float | None = None
    entry_turnover: float = 0.0
    transaction_costs: float = 0.0
    cancellation_reason: str | None = None
    lifecycle_state: str | None = None
    modification_history: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(order_id=self.order_id, symbol=self.symbol, instrument_token=self.instrument_token, strategy_id=self.strategy_id, leg_id=self.leg_id)
        _normalize_timestamp(self, "created_at")
        _normalize_timestamp(self, "updated_at")
        _require_positive(quantity=self.quantity, lot_size=self.lot_size)
        _require_optional_positive(requested_price=self.requested_price, trigger_price=self.trigger_price, fill_price=self.fill_price)
        if not 0 <= self.filled_quantity <= self.quantity:
            raise ValueError("filled_quantity must be between zero and quantity")
        if self.remaining_quantity is None:
            object.__setattr__(self, "remaining_quantity", self.quantity - self.filled_quantity)
        if self.remaining_quantity != self.quantity - self.filled_quantity:
            raise ValueError("remaining_quantity must equal quantity minus filled_quantity")
        _require_optional_positive(average_fill_price=self.average_fill_price)
        _require_non_negative(entry_turnover=self.entry_turnover, transaction_costs=self.transaction_costs)


@dataclass(frozen=True)
class PaperPosition(JsonSerializable):
    """Open or closed multi-leg PAPER strategy position."""

    position_id: str
    opened_at: datetime
    updated_at: datetime
    symbol: str
    instrument_token: str
    strategy_id: str
    strategy_type: StrategyType
    legs: tuple[StrategyLeg, ...]
    side: OrderSide
    quantity: int
    lot_size: int
    status: OrderStatus
    entry_price: float
    current_price: float
    risk_amount: float
    entry_reason: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    transaction_costs: float = 0.0
    slippage: float = 0.0
    closed_at: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(position_id=self.position_id, symbol=self.symbol, instrument_token=self.instrument_token, strategy_id=self.strategy_id, entry_reason=self.entry_reason)
        _normalize_timestamp(self, "opened_at")
        _normalize_timestamp(self, "updated_at")
        _normalize_optional_timestamp(self, "closed_at")
        _require_positive(quantity=self.quantity, lot_size=self.lot_size, entry_price=self.entry_price, current_price=self.current_price)
        _require_optional_positive(exit_price=self.exit_price)
        _require_non_negative(risk_amount=self.risk_amount, transaction_costs=self.transaction_costs, slippage=self.slippage)
        if not self.legs:
            raise ValueError("PaperPosition requires at least one strategy leg")


@dataclass(frozen=True)
class PaperPortfolioState(JsonSerializable):
    """Complete persisted PAPER portfolio accounting state."""

    portfolio_id: str
    timestamp: datetime
    initial_capital: float
    current_equity: float
    available_cash: float
    margin_used: float
    risk_amount: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    transaction_costs: float = 0.0
    slippage: float = 0.0
    positions: tuple[PaperPosition, ...] = ()
    mode: Mode = Mode.PAPER
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(portfolio_id=self.portfolio_id)
        _normalize_timestamp(self, "timestamp")
        _require_non_negative(initial_capital=self.initial_capital, current_equity=self.current_equity, available_cash=self.available_cash, margin_used=self.margin_used, risk_amount=self.risk_amount, transaction_costs=self.transaction_costs, slippage=self.slippage)
        if self.mode is not Mode.PAPER:
            raise ValueError("PaperPortfolioState mode must remain PAPER")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")


@dataclass(frozen=True)
class MarketSnapshot(JsonSerializable):
    """Point-in-time market input for a PAPER decision cycle."""

    snapshot_id: str
    timestamp: datetime
    symbol: str
    instrument_token: str
    last_price: float
    bid: float | None = None
    ask: float | None = None
    available_liquidity: int = 0
    option_chain: OptionChainSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(snapshot_id=self.snapshot_id, symbol=self.symbol, instrument_token=self.instrument_token)
        _normalize_timestamp(self, "timestamp")
        _require_positive(last_price=self.last_price)
        _require_optional_positive(bid=self.bid, ask=self.ask)
        _require_non_negative(available_liquidity=self.available_liquidity)
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError("bid cannot exceed ask")


@dataclass(frozen=True)
class SystemHealthState(JsonSerializable):
    """Health and safety state observed during a PAPER cycle."""

    health_id: str
    timestamp: datetime
    healthy: bool
    market_data_connected: bool
    state_store_available: bool
    kill_switch_active: bool
    consecutive_failures: int = 0
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(health_id=self.health_id)
        _normalize_timestamp(self, "timestamp")
        _require_non_negative(consecutive_failures=self.consecutive_failures)


class ReconciliationOutcome(StrEnum):
    """Possible reconciliation outcomes."""

    CLEAN = "CLEAN"
    EVENTUAL_CONSISTENCY_PENDING = "EVENTUAL_CONSISTENCY_PENDING"
    LOCAL_ORDER_MISSING = "LOCAL_ORDER_MISSING"
    BROKER_ORDER_MISSING = "BROKER_ORDER_MISSING"
    ORDER_STATE_MISMATCH = "ORDER_STATE_MISMATCH"
    FILL_MISMATCH = "FILL_MISMATCH"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    PRICE_MISMATCH = "PRICE_MISMATCH"
    COST_MISMATCH = "COST_MISMATCH"
    POSITION_MISMATCH = "POSITION_MISMATCH"
    UNKNOWN_ORDER = "UNKNOWN_ORDER"
    AMBIGUOUS = "AMBIGUOUS"
    BROKER_UNAVAILABLE = "BROKER_UNAVAILABLE"


ENTRY_BLOCKING_OUTCOMES = {
    ReconciliationOutcome.UNKNOWN_ORDER,
    ReconciliationOutcome.AMBIGUOUS,
    ReconciliationOutcome.LOCAL_ORDER_MISSING,
    ReconciliationOutcome.BROKER_ORDER_MISSING,
    ReconciliationOutcome.ORDER_STATE_MISMATCH,
    ReconciliationOutcome.FILL_MISMATCH,
    ReconciliationOutcome.QUANTITY_MISMATCH,
    ReconciliationOutcome.PRICE_MISMATCH,
    ReconciliationOutcome.COST_MISMATCH,
    ReconciliationOutcome.POSITION_MISMATCH,
    ReconciliationOutcome.BROKER_UNAVAILABLE,
}

EXIT_ALLOWING_OUTCOMES = {
    ReconciliationOutcome.CLEAN,
    ReconciliationOutcome.EVENTUAL_CONSISTENCY_PENDING,
    ReconciliationOutcome.UNKNOWN_ORDER,
    ReconciliationOutcome.AMBIGUOUS,
    ReconciliationOutcome.LOCAL_ORDER_MISSING,
    ReconciliationOutcome.BROKER_ORDER_MISSING,
    ReconciliationOutcome.ORDER_STATE_MISMATCH,
    ReconciliationOutcome.FILL_MISMATCH,
    ReconciliationOutcome.QUANTITY_MISMATCH,
    ReconciliationOutcome.PRICE_MISMATCH,
    ReconciliationOutcome.COST_MISMATCH,
    ReconciliationOutcome.POSITION_MISMATCH,
    ReconciliationOutcome.BROKER_UNAVAILABLE,
}


@dataclass(frozen=True)
class ReconciliationResult(JsonSerializable):
    """Result of comparing persisted PAPER orders, fills, and positions."""

    reconciliation_id: str
    session_id: str
    timestamp: datetime
    outcome: ReconciliationOutcome
    entry_blocked: bool = False
    exits_allowed: bool = True
    affected_order_ids: tuple[str, ...] = ()
    affected_position_ids: tuple[str, ...] = ()
    local_state: dict[str, Any] = field(default_factory=dict)
    broker_state: dict[str, Any] = field(default_factory=dict)
    mismatch_details: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    valid_until: datetime | None = None
    matched_orders: int = 0
    matched_fills: int = 0
    matched_positions: int = 0
    repaired_items: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(reconciliation_id=self.reconciliation_id, session_id=self.session_id)
        _normalize_timestamp(self, "timestamp")
        if isinstance(self.outcome, str):
            object.__setattr__(self, "outcome", ReconciliationOutcome(self.outcome))
        if not isinstance(self.outcome, ReconciliationOutcome):
            raise ValueError("outcome must be a ReconciliationOutcome")
        if self.valid_until is not None:
            _normalize_timestamp(self, "valid_until")
        _require_non_negative(matched_orders=self.matched_orders, matched_fills=self.matched_fills, matched_positions=self.matched_positions)


@dataclass(frozen=True)
class TradingCycleResult(JsonSerializable):
    """Structured result returned by one bounded PAPER trading cycle."""

    cycle_id: str
    started_at: datetime
    completed_at: datetime
    success: bool
    action: str
    portfolio: PaperPortfolioState
    health: SystemHealthState
    reconciliation: ReconciliationResult | None = None
    orders: tuple[PaperOrder, ...] = ()
    fills: tuple[PaperFill, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(cycle_id=self.cycle_id, action=self.action)
        _normalize_timestamp(self, "started_at")
        _normalize_timestamp(self, "completed_at")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")


@dataclass(frozen=True)
class DecisionContext(JsonSerializable):
    """Advisory causal decision context for one PAPER decision cycle."""

    decision_cycle_id: str
    session_id: str
    timestamp: datetime
    market_snapshot_id: str
    feature_snapshot_id: str | None = None
    feature_timestamp: datetime | None = None
    regime: str | None = None
    regime_confidence: float = 0.0
    regime_warnings: tuple[str, ...] = ()
    prediction_probabilities: tuple[float, float, float] | None = None
    prediction_confidence: float = 0.0
    expected_return_percent: float | None = None
    expected_range_low: float | None = None
    expected_range_high: float | None = None
    model_names: tuple[str, ...] = ()
    model_versions: tuple[str, ...] = ()
    volatility_forecast: float | None = None
    volatility_model_requested: str | None = None
    volatility_model_used: str | None = None
    convergence_status: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    expected_expiry_price: float | None = None
    expected_move: float | None = None
    range_68: tuple[float, float] | None = None
    range_95: tuple[float, float] | None = None
    tail_risk_score: float | None = None
    news_sentiment: float | None = None
    event_risk_level: str | None = None
    liquidity_score: float | None = None
    entry_context_valid: bool = False
    invalid_reason: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        _require_text(decision_cycle_id=self.decision_cycle_id, session_id=self.session_id, market_snapshot_id=self.market_snapshot_id)
        _normalize_timestamp(self, "timestamp")
        if self.feature_timestamp is not None:
            _normalize_timestamp(self, "feature_timestamp")
        if self.feature_timestamp is not None and self.feature_timestamp > self.timestamp:
            raise ValueError("feature_timestamp cannot exceed timestamp")
        if self.expected_range_low is not None and self.expected_range_high is not None:
            if self.expected_range_low > self.expected_range_high:
                raise ValueError("expected_range_low cannot exceed expected_range_high")
        if self.range_68 is not None:
            if not isinstance(self.range_68, tuple) or len(self.range_68) != 2:
                raise ValueError("range_68 must be a tuple of two floats")
            if self.range_68[0] > self.range_68[1]:
                raise ValueError("range_68 lower bound cannot exceed upper bound")
        if self.range_95 is not None:
            if not isinstance(self.range_95, tuple) or len(self.range_95) != 2:
                raise ValueError("range_95 must be a tuple of two floats")
            if self.range_95[0] > self.range_95[1]:
                raise ValueError("range_95 lower bound cannot exceed upper bound")
        _require_non_negative(regime_confidence=self.regime_confidence, prediction_confidence=self.prediction_confidence, tail_risk_score=self.tail_risk_score or 0.0)
        if self.liquidity_score is not None:
            _require_non_negative(liquidity_score=self.liquidity_score)
        if self.news_sentiment is not None:
            if not -1.0 <= self.news_sentiment <= 1.0:
                raise ValueError("news_sentiment must be between -1.0 and 1.0")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _restore_value(expected: Any, value: Any) -> Any:
    origin = get_origin(expected)
    args = get_args(expected)
    if value is None:
        return None
    if origin in (Union, UnionType):
        target = next((item for item in args if item is not type(None)), Any)
        return _restore_value(target, value)
    if origin in (tuple, list):
        target = args[0] if args else Any
        converted = [_restore_value(target, item) for item in value]
        return tuple(converted) if origin is tuple else converted
    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {key: _restore_value(value_type, item) for key, item in value.items()}
    if expected is Any:
        return value
    if expected is datetime:
        return ensure_ist(datetime.fromisoformat(value))
    if expected is date:
        return date.fromisoformat(value)
    if isinstance(expected, type) and issubclass(expected, Enum):
        return expected(value)
    if isinstance(expected, type) and is_dataclass(expected):
        if issubclass(expected, JsonSerializable):
            return expected.from_dict(value)
        hints = get_type_hints(expected)
        return expected(**{item.name: _restore_value(hints[item.name], value[item.name]) for item in fields(expected) if item.name in value})
    return value


def _normalize_timestamp(instance: Any, name: str) -> None:
    value = getattr(instance, name)
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    object.__setattr__(instance, name, ensure_ist(value))


def _normalize_optional_timestamp(instance: Any, name: str) -> None:
    if getattr(instance, name) is not None:
        _normalize_timestamp(instance, name)


def _require_text(**values: str) -> None:
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _require_optional_positive(**values: float | None) -> None:
    _require_positive(**{name: value for name, value in values.items() if value is not None})


def _require_non_negative(**values: float) -> None:
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative")