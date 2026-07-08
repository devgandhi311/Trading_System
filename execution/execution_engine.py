"""Execution engine orchestrating approved strategy plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Callable, Mapping

from execution.order_pricing import build_profit_booking_order, build_short_option_stop_loss_order
from execution.order_router import OrderBasket, OrderRouter
from execution.order_validator import OrderValidator
from execution.squareoff import SquareoffManager, SquareoffResult
from execution.trade_state import ExecutionTradeState
from models import Mode, OrderResponse, OrderSide, OrderStatus, RiskDecision, StrategyPlan
from models.order_models import OrderRequest
from risk import RiskManager
from utils.config_loader import ConfigLoader
from utils.exceptions import ExecutionError, RiskRejectedError
from utils.time_utils import ensure_ist, now_ist, should_force_squareoff

LOGGER = logging.getLogger(__name__)
QuoteProvider = Callable[[str], Any]


@dataclass(frozen=True)
class ExecutionResult:
    """Result of an execution engine action."""

    approved: bool
    status: str
    message: str
    risk_decision: RiskDecision | None = None
    entry_orders: tuple[OrderResponse, ...] = ()
    stop_loss_orders: tuple[OrderResponse, ...] = ()
    profit_booking_orders: tuple[OrderResponse, ...] = ()
    cancelled_orders: tuple[OrderResponse, ...] = ()
    squareoff_result: SquareoffResult | None = None
    state: ExecutionTradeState | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """Coordinate risk-approved strategy execution."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        risk_manager: RiskManager | None = None,
        order_router: OrderRouter | None = None,
        order_validator: OrderValidator | None = None,
    ) -> None:
        """Create execution engine with safe defaults."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.risk_manager = risk_manager or RiskManager(config_loader=self.config_loader)
        self.order_router = order_router or OrderRouter(config_loader=self.config_loader)
        self.order_validator = order_validator or OrderValidator(config_loader=self.config_loader)
        self.squareoff_manager = SquareoffManager(order_router=self.order_router)

    def execute_strategy(
        self,
        plan: StrategyPlan,
        *,
        quote_provider: QuoteProvider,
        account_context: Mapping[str, Any] | None = None,
        market_context: Mapping[str, Any] | None = None,
        manual_squareoff: bool = False,
        timestamp: datetime | None = None,
    ) -> ExecutionResult:
        """Execute an approved StrategyPlan through the safe order flow."""

        evaluated_at = ensure_ist(timestamp or now_ist())
        self._validate_live_gate()
        self._validate_strategy_plan(plan)
        state = ExecutionTradeState.from_plan(plan)
        news_level = str((market_context or {}).get("event_risk_level") or plan.metadata.get("event_risk_level", "")).upper()
        if news_level in {"HIGH", "EXTREME"}:
            return ExecutionResult(
                approved=False,
                status="news_risk_rejected",
                message=f"News event risk {news_level} blocks order placement.",
                state=state,
            )

        risk_decision = self.risk_manager.validate(
            plan,
            account_context=account_context,
            market_context=market_context,
            timestamp=evaluated_at,
        )
        if not risk_decision.approved:
            return ExecutionResult(
                approved=False,
                status="risk_rejected",
                message=risk_decision.rejection_reason or risk_decision.reason,
                risk_decision=risk_decision,
                state=state,
                warnings=risk_decision.warnings,
            )

        if not self._margin_approved(risk_decision, account_context or {}):
            return ExecutionResult(
                approved=False,
                status="margin_rejected",
                message="Available margin is insufficient.",
                risk_decision=risk_decision,
                state=state,
            )

        basket = self.order_router.prepare_entry_basket(plan)
        self._validate_all_orders(basket, quote_provider)
        entry_responses = self.order_router.place_basket(basket, quote_provider)
        for response in entry_responses:
            state.apply_order_response(response)

        cancelled = self.cancel_unfilled_after_timeout(
            basket,
            entry_responses,
            submitted_at=evaluated_at,
            now=evaluated_at,
        )

        squareoff_result = self.squareoff_manager.handle_partial_fill(state, quote_provider=quote_provider)
        if manual_squareoff and state.filled_legs():
            squareoff_result = self.squareoff_manager.manual_squareoff(state, quote_provider=quote_provider)

        stop_loss_orders: tuple[OrderResponse, ...] = ()
        profit_orders: tuple[OrderResponse, ...] = ()
        if state.all_entries_filled():
            stop_loss_orders, profit_orders = self.place_protective_orders(
                plan,
                state,
                quote_provider=quote_provider,
            )

        exit_reason = self.evaluate_exit_conditions(
            plan,
            state,
            account_context=account_context,
            manual_squareoff=manual_squareoff,
            timestamp=evaluated_at,
        )
        if exit_reason and state.filled_legs() and squareoff_result is None:
            squareoff_result = self.squareoff_manager.squareoff_filled_legs(
                state,
                quote_provider=quote_provider,
                reason=exit_reason,
            )

        LOGGER.info(
            "Execution flow completed",
            extra={"strategy_id": plan.strategy_id, "entries": len(entry_responses), "status": "submitted"},
        )
        return ExecutionResult(
            approved=True,
            status="submitted",
            message="Execution flow completed.",
            risk_decision=risk_decision,
            entry_orders=entry_responses,
            stop_loss_orders=stop_loss_orders,
            profit_booking_orders=profit_orders,
            cancelled_orders=cancelled,
            squareoff_result=squareoff_result,
            state=state,
            warnings=risk_decision.warnings,
        )

    def cancel_unfilled_after_timeout(
        self,
        basket: OrderBasket,
        responses: tuple[OrderResponse, ...],
        *,
        submitted_at: datetime,
        now: datetime | None = None,
    ) -> tuple[OrderResponse, ...]:
        """Cancel unfilled entry orders once timeout is reached."""

        cancelled: list[OrderResponse] = []
        order_by_symbol = {order.symbol: order for order in basket.orders}
        for response in responses:
            order = order_by_symbol.get(response.symbol)
            if order is None or response.status == OrderStatus.FILLED:
                continue
            if self.order_validator.should_cancel_unfilled(order, submitted_at=submitted_at, now=now):
                cancelled.append(self.order_router.cancel_order(response.order_id))
        return tuple(cancelled)

    def place_protective_orders(
        self,
        plan: StrategyPlan,
        state: ExecutionTradeState,
        *,
        quote_provider: QuoteProvider,
    ) -> tuple[tuple[OrderResponse, ...], tuple[OrderResponse, ...]]:
        """Place stop-loss and profit-booking orders for filled short-option legs."""

        stop_loss_percent = float(plan.metadata.get("stop_loss_percent", 25.0))
        sl_buffer = float(plan.metadata.get("sl_limit_buffer", 1.0))
        target_buffer = float(plan.metadata.get("target_price_buffer", 0.0))
        stop_orders: list[OrderResponse] = []
        profit_orders: list[OrderResponse] = []
        for leg in state.filled_legs():
            if leg.side != OrderSide.SELL or leg.average_price is None:
                continue
            stop_order = build_short_option_stop_loss_order(
                symbol=leg.symbol,
                quantity=leg.filled_quantity,
                entry_price=leg.average_price,
                stop_loss_percent=stop_loss_percent,
                buffer=sl_buffer,
                strategy_id=plan.strategy_id,
            )
            profit_order = build_profit_booking_order(
                symbol=leg.symbol,
                quantity=leg.filled_quantity,
                quote=quote_provider(leg.symbol),
                buffer=target_buffer,
                strategy_id=plan.strategy_id,
            )
            stop_response = self.order_router.place_order(stop_order, quote_provider)
            profit_response = self.order_router.place_order(profit_order, quote_provider)
            stop_orders.append(stop_response)
            profit_orders.append(profit_response)
            state.stop_loss_order_ids.append(stop_response.order_id)
            state.profit_order_ids.append(profit_response.order_id)
        return tuple(stop_orders), tuple(profit_orders)

    def monitor_live_pnl(
        self,
        state: ExecutionTradeState,
        *,
        quote_provider: QuoteProvider,
    ) -> float:
        """Update live P&L from latest quotes and return unrealized P&L."""

        for leg in state.filled_legs():
            quote = quote_provider(leg.symbol)
            state.update_last_price(leg.symbol, _quote_ltp(quote))
        return state.unrealized_pnl

    def evaluate_exit_conditions(
        self,
        plan: StrategyPlan,
        state: ExecutionTradeState,
        *,
        account_context: Mapping[str, Any] | None = None,
        manual_squareoff: bool = False,
        regime_changed: bool = False,
        timestamp: datetime | None = None,
    ) -> str | None:
        """Evaluate target, stop, regime change, time, kill switch, daily loss, and manual exits."""

        account = dict(account_context or {})
        if manual_squareoff or state.manual_squareoff_requested:
            return "manual_squareoff"
        if self.risk_manager.kill_switch.active:
            return "kill_switch"
        if regime_changed:
            return "regime_change"
        if should_force_squareoff(timestamp or now_ist(), force_squareoff_time=self.config.settings.force_squareoff_time):
            return "time_exit"
        daily_loss_limit = self.config.risk_limits.capital * self.config.risk_limits.max_daily_loss_percent / 100.0
        if float(account.get("daily_pnl", 0.0)) <= -daily_loss_limit:
            return "daily_loss"
        target_profit = _optional_float(plan.metadata.get("target_profit"))
        stop_loss = _optional_float(plan.metadata.get("stop_loss"))
        pnl = state.realized_pnl + state.unrealized_pnl
        if target_profit is not None and pnl >= target_profit:
            return "target"
        if stop_loss is not None and pnl <= -abs(stop_loss):
            return "stop_loss"
        return None

    def _validate_live_gate(self) -> None:
        if self.config.settings.mode == Mode.LIVE and not self.config.settings.live_trading_enabled:
            raise ExecutionError("LIVE orders require mode=LIVE and live_trading_enabled=true")

    @staticmethod
    def _validate_strategy_plan(plan: StrategyPlan) -> None:
        if not isinstance(plan, StrategyPlan):
            raise ExecutionError("plan must be a StrategyPlan")
        if not plan.legs:
            raise ExecutionError("StrategyPlan must include legs")
        if plan.max_loss is None or plan.max_loss <= 0:
            raise RiskRejectedError("StrategyPlan max_loss must be defined before execution")
        for leg in plan.legs:
            if leg.quantity <= 0:
                raise ExecutionError("All strategy legs must have positive quantity")
            if leg.limit_price is None or leg.limit_price <= 0:
                raise ExecutionError("Entry legs must include positive limit_price")

    def _validate_all_orders(self, basket: OrderBasket, quote_provider: QuoteProvider) -> None:
        for order in basket.orders:
            self.order_validator.validate_order_price(order, quote_provider(order.symbol))

    @staticmethod
    def _margin_approved(risk_decision: RiskDecision, account_context: Mapping[str, Any]) -> bool:
        available_margin = account_context.get("available_margin")
        if available_margin is None:
            return True
        return float(available_margin) >= float(risk_decision.margin_required)


def _quote_ltp(quote: Any) -> float:
    if isinstance(quote, dict):
        return float(quote.get("ltp", quote.get("last_price")))
    ltp = getattr(quote, "ltp", None)
    if ltp is None:
        ltp = getattr(quote, "last_price", None)
    return float(ltp)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
