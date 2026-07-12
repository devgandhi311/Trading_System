"""Shared runtime state models for PAPER trading only."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from types import UnionType
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

from models.common import Mode, OrderSide, OrderStatus, OrderType, StrategyType
from models.market_models import OptionChainSnapshot
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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(order_id=self.order_id, symbol=self.symbol, instrument_token=self.instrument_token, strategy_id=self.strategy_id, leg_id=self.leg_id)
        _normalize_timestamp(self, "created_at")
        _normalize_timestamp(self, "updated_at")
        _require_positive(quantity=self.quantity, lot_size=self.lot_size)
        _require_optional_positive(requested_price=self.requested_price, trigger_price=self.trigger_price, fill_price=self.fill_price)
        if not 0 <= self.filled_quantity <= self.quantity:
            raise ValueError("filled_quantity must be between zero and quantity")


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


@dataclass(frozen=True)
class ReconciliationResult(JsonSerializable):
    """Result of comparing persisted PAPER orders, fills, and positions."""

    reconciliation_id: str
    timestamp: datetime
    reconciled: bool
    matched_orders: int = 0
    matched_fills: int = 0
    matched_positions: int = 0
    repaired_items: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(reconciliation_id=self.reconciliation_id)
        _normalize_timestamp(self, "timestamp")
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
