"""Autonomous, fail-closed PAPER trading orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import json
import hashlib
import logging
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

import pandas as pd

from broker.paper_broker import PaperBroker
from models.common import OrderStatus, RegimeType
from models.market_models import PaperMarketSnapshot
from models.order_models import OrderRequest
from models.paper_models import DecisionContext
from models.prediction_models import PredictionResult
from models.strategy_models import StrategyPlan
from risk.risk_manager import RiskManager
from runtime.market_snapshot_validator import MarketSnapshotValidator
from runtime.paper_decision_context import PaperDecisionContextService
from runtime.paper_portfolio_service import PaperPortfolioService
from runtime.reconciliation import PaperReconciliationService
from runtime.session_scheduler import SessionScheduler
from storage.database import OperationalDatabase
from storage.repositories import OperationalRepository
from strategies.strategy_selector import StrategySelector
from utils.config_loader import ConfigLoader
from utils.time_utils import ensure_ist, now_ist
from runtime.dashboard_projection import AuthoritativeDashboardProjector
from runtime.position_monitor import PositionMonitor, PositionQuote
from runtime.recovery_manager import RecoveryManager
from runtime.end_of_day import EndOfDayController

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperCycleInputs:
    """Point-in-time inputs supplied to one PAPER cycle."""

    market_snapshot: PaperMarketSnapshot
    historical_bars: pd.DataFrame
    news_state: Any = None


class PaperTradingEngine:
    """Run bounded PAPER cycles with durable idempotency and restart recovery."""

    def __init__(
        self,
        config_loader: ConfigLoader | None = None,
        *,
        database: OperationalDatabase | None = None,
        state_dir: str | Path = "data/live",
        input_provider: Callable[[], PaperCycleInputs] | None = None,
        scheduler: SessionScheduler | None = None,
        validator: MarketSnapshotValidator | None = None,
        context_service: PaperDecisionContextService | None = None,
        strategy_selector: StrategySelector | None = None,
        risk_manager_factory: Callable[[PaperPortfolioService], RiskManager] | None = None,
        broker_factory: Callable[[str], PaperBroker] | None = None,
        reconciliation_factory: Callable[[str], PaperReconciliationService] | None = None,
        prediction_resolver: Callable[[DecisionContext, PaperMarketSnapshot], PredictionResult] | None = None,
        clock: Callable[[], datetime] = now_ist,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        if str(self.config.settings.mode).upper() != "PAPER":
            raise RuntimeError("PaperTradingEngine requires PAPER mode")
        if self.config.settings.live_trading_enabled or self.config.settings.live_order_submission_enabled:
            raise RuntimeError("LIVE flags must be disabled for PaperTradingEngine")

        self.database = database or OperationalDatabase(Path(state_dir) / "trading_state.db")
        self.database.initialize()
        self.repository = OperationalRepository(self.database)
        self.state_dir = Path(state_dir)
        self.input_provider = input_provider
        self.clock = clock
        self.poll_interval_seconds = max(0.1, poll_interval_seconds)
        self.stop_event = Event()
        settings = self.config.settings
        self.scheduler = scheduler or SessionScheduler(
            self.database,
            mode="PAPER",
            clock=clock,
            market_start=settings.market_start_time,
            market_end=settings.market_end_time,
            no_new_entry_cutoff=settings.no_new_trade_after_time,
            squareoff_start=settings.force_squareoff_time,
        )
        self.validator = validator or MarketSnapshotValidator()
        self.context_service = context_service or PaperDecisionContextService(
            self.repository, self.config_loader, clock=clock
        )
        self.strategy_selector = strategy_selector or StrategySelector(config_loader=self.config_loader)
        self._risk_manager_factory = risk_manager_factory or (
            lambda portfolio: RiskManager(
                config_loader=self.config_loader,
                portfolio_service=portfolio,
                database=self.database,
            )
        )
        limits = self.config.risk_limits
        self._broker_factory = broker_factory or (
            lambda session_id: PaperBroker(
                self.repository,
                session_id=session_id,
                clock=clock,
                slippage_percent=limits.max_slippage_percent,
                emergency_market_exit_allowed=limits.emergency_market_exit_allowed,
            )
        )
        self._reconciliation_factory = reconciliation_factory or (
            lambda session_id: PaperReconciliationService(self.repository, session_id)
        )
        self._prediction_resolver = prediction_resolver or self._prediction_from_context
        self.dashboard_projector = AuthoritativeDashboardProjector(self.repository, self.state_dir, clock=clock)
        self._cycle_started_at: datetime | None = None
        self._duplicates_prevented_cycle = 0
        self._startup_recovered = False
        self._last_inputs: PaperCycleInputs | None = None

    def request_shutdown(self) -> None:
        """Request graceful termination after the current bounded cycle."""

        self.stop_event.set()

    def run(self, *, continuous: bool = False, max_cycles: int | None = None) -> int:
        """Run one or multiple cycles, returning the number attempted."""

        attempted = 0
        try:
            while not self.stop_event.is_set():
                self.run_cycle()
                attempted += 1
                if not continuous or (max_cycles is not None and attempted >= max_cycles):
                    break
                self.stop_event.wait(self.poll_interval_seconds)
        except KeyboardInterrupt:
            self.request_shutdown()
        finally:
            LOGGER.info("PAPER engine stopped", extra={"cycles_attempted": attempted})
        return attempted

    def run_cycle(self) -> str:
        """Execute one ordered PAPER transaction boundary and return its action."""

        self._ensure_recovery()
        started_at = ensure_ist(self.clock())
        self._cycle_started_at = started_at
        self._duplicates_prevented_cycle = 0
        session = self.scheduler.open_or_resume_session()
        self.scheduler.poll()
        portfolio = PaperPortfolioService(self.database, initial_capital=self.config.risk_limits.capital, session_id=session.id)
        portfolio.initialize()
        if getattr(self.scheduler, "exits_allowed", lambda: False)():
            self._position_monitor(session.id, portfolio).monitor_once()
        if getattr(self.scheduler, "squareoff_required", lambda: False)():
            result = self._end_of_day(session.id, portfolio).run()
            self._project(session.id, action=f"EOD_{result.status}", context=None, risk=None)
            return f"EOD_{result.status}"
        if not self.scheduler.entry_allowed():
            action = "SCHEDULER_BLOCKED"
            self._project(session.id, action=action, context=None, risk=None)
            return action

        can_trade, reason = portfolio.can_trade()
        reconciliation = self._reconciliation_factory(session.id)
        if not can_trade or reconciliation.has_unresolved_blocking_reconciliation():
            action = "PERSISTED_STATE_BLOCKED"
            self._project(session.id, action=action, context=None, risk={"reason": reason})
            return action
        if self.input_provider is None:
            raise RuntimeError("No PAPER market input provider configured")

        inputs = self.input_provider()
        self._last_inputs = inputs
        validation = self.validator.validate(inputs.market_snapshot, current_time=started_at)
        if not validation.valid or not validation.entry_allowed:
            action = "MARKET_DATA_BLOCKED"
            self._project(session.id, action=action, context=None, risk={"reasons": validation.blocking_reasons})
            return action
        self.repository.save_market_snapshot(
            snapshot_id=inputs.market_snapshot.snapshot_id, session_id=session.id,
            symbol=inputs.market_snapshot.underlying,
            instrument_token=inputs.market_snapshot.spot_instrument_token,
            captured_at=inputs.market_snapshot.timestamp,
            payload={"spot_price": inputs.market_snapshot.spot_price,
                     "timestamp": inputs.market_snapshot.timestamp.isoformat(),
                     "source": inputs.market_snapshot.metadata.get("source"),
                     "option_contract_count": len(inputs.market_snapshot.option_chain.contracts)},
        )

        context = self.context_service.build_context(
            session,
            inputs.market_snapshot,
            inputs.historical_bars,
            portfolio.get_state(),
            inputs.news_state,
            decision_time=started_at,
        )
        if not context.entry_context_valid:
            action = "DECISION_BLOCKED"
            self._project(session.id, action=action, context=context, risk=None)
            return action

        prediction = self._prediction_resolver(context, inputs.market_snapshot)
        plan = self.strategy_selector.select(
            prediction,
            inputs.market_snapshot.option_chain,
            features={"liquidity_score": context.liquidity_score},
            risk_context={"event_risk_level": context.event_risk_level},
        )
        if not plan.legs or plan.metadata.get("blocked") or not plan.metadata.get("valid", True):
            action = "NO_TRADE"
            self._project(session.id, action=action, context=context, risk=None)
            return action

        risk = self._risk_manager_factory(portfolio).validate(
            plan,
            market_context={"event_risk_level": context.event_risk_level},
            regime=context.regime,
            prediction=prediction,
            timestamp=started_at,
        )
        strategy_instance_key = self._strategy_instance_key(session.id, context, plan, inputs.market_snapshot)
        risk_id = hashlib.sha256(f"risk|{strategy_instance_key}".encode()).hexdigest()
        self.repository.save_risk_decision(
            risk_id=risk_id, session_id=session.id, decision_cycle_id=context.decision_cycle_id,
            approved=risk.approved, reason=risk.reason, decided_at=risk.timestamp,
            metadata={"snapshot_id": inputs.market_snapshot.snapshot_id,
                      "strategy_instance_key": strategy_instance_key,
                      "strategy_type": plan.strategy_type.value, "checks": risk.checks,
                      "limits": risk.metadata, "quantity": risk.allowed_quantity,
                      "risk_used": risk.risk_used, "margin_required": risk.margin_required},
        )
        if not risk.approved:
            action = "RISK_BLOCKED"
            self._project(session.id, action=action, context=context, risk=risk)
            return action

        broker = self._broker_factory(session.id)
        orders = []
        newly_reserved = False
        risk_reserved = False
        for index, leg in enumerate(plan.legs):
            contract = self._contract(inputs.market_snapshot, leg.symbol)
            # Snapshot + strategy + leg identifies the immutable economic intent
            # across retries and process restarts; decision wall-clock time does not.
            key = hashlib.sha256(f"order|{strategy_instance_key}|{index}".encode()).hexdigest()
            existing = self.repository.get_order_intent(key)
            if existing is not None:
                self._duplicates_prevented_cycle += 1
            elif not risk_reserved:
                portfolio.reserve_pending_order_risk(risk.risk_used, risk.margin_required)
                risk_reserved = True
            request = OrderRequest(
                symbol=leg.symbol,
                side=leg.side,
                quantity=min(leg.quantity, risk.allowed_quantity or leg.quantity),
                order_type=leg.order_type,
                timestamp=started_at,
                strategy_id=plan.strategy_id,
                limit_price=leg.limit_price,
                trigger_price=leg.trigger_price,
                metadata={"idempotency_key": key, "leg_id": str(index), "instrument_token": contract.instrument_token},
            )
            # submit_order performs the durable reservation before any simulated submission.
            quote = self._paper_quote(inputs.market_snapshot, contract)
            orders.append(broker.submit_order(request, quote, instrument_token=contract.instrument_token, lot_size=contract.lot_size))
            newly_reserved = newly_reserved or existing is None

        for order in orders:
            if order.filled_quantity and order.average_fill_price is not None:
                contract = self._contract(inputs.market_snapshot, order.symbol)
                signed_quantity = order.filled_quantity if order.side.value == "BUY" else -order.filled_quantity
                self.repository.save_position(
                    position_id=f"paper:{order.order_id}",
                    session_id=session.id,
                    symbol=order.symbol,
                    quantity=signed_quantity,
                    average_price=order.average_fill_price,
                    strategy_id=plan.strategy_id,
                    metadata={"strategy_instance_key": strategy_instance_key,
                              "instrument_token": contract.instrument_token, "expiry": contract.expiry.isoformat(),
                              "option_type": contract.option_type, "strike": contract.strike, "side": order.side.value,
                              "entry_timestamp": order.updated_at.isoformat(), "risk_amount": risk.risk_used,
                              "margin_required": risk.margin_required, "decision_id": context.decision_cycle_id,
                              "order_intent_id": order.order_id, "snapshot_id": inputs.market_snapshot.snapshot_id,
                              "stop_loss": plan.metadata.get("stop_loss"), "target": plan.metadata.get("target_price"),
                              "time_exit_at": plan.metadata.get("time_exit_at")},
                )

        if risk_reserved and any(order.status in {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.SUBMITTED} for order in orders):
            pass
        elif risk_reserved and any(order.status == OrderStatus.FILLED for order in orders):
            portfolio.release_pending_order_risk(risk.risk_used, risk.margin_required)
            portfolio.open_strategy(risk_amount=risk.risk_used, margin_required=risk.margin_required)
        elif risk_reserved:
            portfolio.release_pending_order_risk(risk.risk_used, risk.margin_required)

        all_orders = [broker.get_order(row.idempotency_key) for row in self.repository.session_orders(session.id)]
        result = reconciliation.reconcile(
            lambda: {order.order_id: order for order in all_orders},
            lambda: {order.order_id: order.fills for order in all_orders},
            lambda: {row.id: row for row in self.repository.open_positions()},
            now=ensure_ist(self.clock()),
        )
        action = "ORDERS_PROCESSED" if not result.entry_blocked else "RECONCILIATION_BLOCKED"
        self._project(session.id, action=action, context=context, risk=risk)
        return action

    def close(self) -> None:
        """Release database resources after shutdown."""

        self.database.dispose()

    def _ensure_recovery(self) -> None:
        if self._startup_recovered: return
        latest = self.repository.get_latest_session()
        if latest is not None and latest.status not in {"CLOSED", "HOLIDAY"}:
            factory = lambda sid: PaperPortfolioService(self.database, initial_capital=self.config.risk_limits.capital, session_id=sid)
            recovery = RecoveryManager(self.repository, self.scheduler, factory, self._reconciliation_factory,
                lambda sid, portfolio: self._position_monitor(sid, portfolio),
                broker_order_provider=lambda: self._paper_orders(latest.id),
                broker_fill_provider=lambda: self._paper_fills(latest.id),
                broker_position_provider=lambda: {row.id: row for row in self.repository.open_positions()})
            result = recovery.recover()
            if result.status == "BLOCKED": raise RuntimeError("Startup recovery unresolved: " + "; ".join(result.blockers))
        self._startup_recovered = True

    def _paper_orders(self, session_id: str) -> dict[str, Any]:
        broker = self._broker_factory(session_id)
        return {row.idempotency_key: broker.get_order(row.idempotency_key) for row in self.repository.session_orders(session_id)}

    def _paper_fills(self, session_id: str) -> dict[str, Any]:
        return {key: order.fills for key, order in self._paper_orders(session_id).items()}

    def _position_monitor(self, session_id: str, portfolio: PaperPortfolioService) -> PositionMonitor:
        def quote(symbol: str) -> PositionQuote:
            if self._last_inputs is None: raise RuntimeError("No validated quote available for position monitoring")
            contract = next((c for c in self._last_inputs.market_snapshot.option_chain.contracts if c.symbol == symbol), None)
            if contract is None: raise RuntimeError(f"No validated quote for {symbol}")
            price = contract.last_price or ((contract.bid + contract.ask) / 2 if contract.bid and contract.ask else 0)
            return PositionQuote(symbol, float(price), contract.quote_timestamp or self._last_inputs.market_snapshot.timestamp)
        def submit(position, exit_id, reason, emergency):
            from models.common import OrderSide, OrderType
            request = OrderRequest(symbol=position.symbol, side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                quantity=abs(position.quantity), order_type=OrderType.MARKET if emergency else OrderType.LIMIT,
                timestamp=ensure_ist(self.clock()), strategy_id=position.strategy_id,
                limit_price=None if emergency else quote(position.symbol).price,
                metadata={"idempotency_key": exit_id, "emergency_exit": emergency,
                          "instrument_token": position.metadata_json.get("instrument_token"), "leg_id": "exit"})
            q = quote(position.symbol)
            from models.paper_models import MarketSnapshot
            market = MarketSnapshot(exit_id, q.timestamp, q.symbol, str(position.metadata_json.get("instrument_token") or "missing"),
                                    q.price, q.price, q.price, abs(position.quantity))
            return self._broker_factory(session_id).submit_order(request, market,
                instrument_token=str(position.metadata_json.get("instrument_token") or "missing"), lot_size=1)
        return PositionMonitor(self.repository, portfolio, session_id=session_id, quote_provider=quote,
                               exit_submitter=submit, clock=self.clock)

    def _end_of_day(self, session_id: str, portfolio: PaperPortfolioService) -> EndOfDayController:
        broker = self._broker_factory(session_id); reconciliation = self._reconciliation_factory(session_id)
        return EndOfDayController(self.repository, portfolio, self._position_monitor(session_id, portfolio), session_id=session_id,
            reconciliation_runner=lambda: reconciliation.reconcile(lambda: self._paper_orders(session_id),
                lambda: self._paper_fills(session_id), lambda: {row.id: row for row in self.repository.open_positions()}),
            cancel_order=lambda key: broker.cancel_order(key), dashboard_dir=self.state_dir, clock=self.clock)

    def _project(self, session_id: str, *, action: str, context: Any, risk: Any) -> None:
        """Persist cycle timing, then rebuild projections only from SQLite."""
        completed = ensure_ist(self.clock())
        elapsed_ms = max(0.0, (completed - (self._cycle_started_at or completed)).total_seconds() * 1000.0)
        session = self.repository.get_trading_session(session_id)
        metadata = dict(session.metadata_json) if session else {}
        samples = list(metadata.get("qualification_cycle_observations", []))
        samples.append({"completed_at": completed.isoformat(), "action": action, "latency_ms": elapsed_ms,
                        "duplicates_prevented": self._duplicates_prevented_cycle})
        self.repository.update_session_metadata(session_id, qualification_cycle_observations=samples[-10000:])
        self.dashboard_projector.project(session_id, action=action)

    @staticmethod
    def _jsonable(value: Any) -> dict[str, Any]:
        if hasattr(value, "to_dict"):
            return dict(value.to_dict())
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Mapping):
            return dict(value)
        return {"value": str(value)}

    @staticmethod
    def _contract(snapshot: PaperMarketSnapshot, symbol: str) -> Any:
        for contract in snapshot.option_chain.contracts:
            if contract.symbol == symbol:
                if not contract.instrument_token:
                    raise RuntimeError(f"Missing instrument token for {symbol}")
                return contract
        raise RuntimeError(f"Strategy leg {symbol} is absent from validated option chain")

    @staticmethod
    def _paper_quote(snapshot: PaperMarketSnapshot, contract: Any) -> Any:
        from models.paper_models import MarketSnapshot

        return MarketSnapshot(
            snapshot_id=f"{snapshot.snapshot_id}:{contract.instrument_token}",
            timestamp=contract.quote_timestamp or snapshot.timestamp,
            symbol=contract.symbol,
            instrument_token=contract.instrument_token,
            bid=contract.bid,
            ask=contract.ask,
            last_price=contract.last_price or (float(contract.bid) + float(contract.ask)) / 2.0,
            available_liquidity=contract.available_quantity or contract.lot_size,
        )

    @staticmethod
    def _prediction_from_context(context: DecisionContext, snapshot: PaperMarketSnapshot) -> PredictionResult:
        probabilities = context.prediction_probabilities or (0.0, 0.0, 0.0)
        labels = ("bullish", "bearish", "sideways")
        direction = labels[max(range(3), key=lambda i: probabilities[i])]
        try:
            regime = RegimeType(context.regime or RegimeType.UNKNOWN.value)
        except ValueError:
            regime = RegimeType.UNKNOWN
        expected_move_percent = 0.0
        if context.expected_move and snapshot.spot_price > 0:
            expected_move_percent = context.expected_move / snapshot.spot_price * 100.0
        return PredictionResult(
            symbol=snapshot.underlying,
            underlying=snapshot.underlying,
            timestamp=context.timestamp,
            predicted_direction=direction,
            confidence=context.prediction_confidence,
            confidence_score=context.prediction_confidence,
            horizon_minutes=375,
            regime=regime,
            bullish_probability=probabilities[0],
            bearish_probability=probabilities[1],
            sideways_probability=probabilities[2],
            expected_return_percent=context.expected_return_percent,
            expected_move_percent=expected_move_percent,
            expected_move_points=context.expected_move,
            expected_expiry_price=context.expected_expiry_price,
            expected_range_low=context.expected_range_low,
            expected_range_high=context.expected_range_high,
            prediction_source="PERSISTED_DECISION_CONTEXT",
            trade_allowed=context.entry_context_valid,
            reason=context.invalid_reason or "Decision context permits strategy selection.",
            metadata={"event_risk_level": context.event_risk_level},
        )

    @staticmethod
    def _strategy_instance_key(session_id: str, context: DecisionContext, plan: StrategyPlan,
                               snapshot: PaperMarketSnapshot) -> str:
        legs = sorted(({"symbol": leg.symbol, "side": leg.side.value, "quantity": leg.quantity,
                       "order_type": leg.order_type.value, "limit_price": leg.limit_price,
                       "trigger_price": leg.trigger_price} for leg in plan.legs), key=lambda item: json.dumps(item, sort_keys=True))
        payload = {"mode": "PAPER", "session_id": session_id, "snapshot_id": snapshot.snapshot_id,
                   "strategy_type": plan.strategy_type.value,
                   "underlying": plan.symbol, "legs": legs}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
