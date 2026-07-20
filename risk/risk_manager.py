"""Central risk manager for strategy approval with persistence support."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Mapping

from models import PredictionResult, RegimeResult, RegimeType, RiskDecision, StrategyPlan
from models.risk_models import PortfolioAccounting
from risk.daily_loss_guard import DailyLossGuard
from risk.exposure_limits import ExposureLimits
from risk.kill_switch import KillSwitch
from risk.position_sizing import PositionSizer
from risk.slippage_model import SlippageModel
from risk.trade_loss_guard import TradeLossGuard
from runtime.paper_portfolio_service import PaperPortfolioService
from storage.database import OperationalDatabase
from utils.config_loader import ConfigLoader
from utils.time_utils import can_place_new_trade, ensure_ist, is_market_open, now_ist

LOGGER = logging.getLogger(__name__)


class RiskManager:
    """Validate strategy plans against configured risk limits.

    Integrates with PaperPortfolioService for authoritative current-equity
    and risk state, and with KillSwitch backed by persistent storage.
    """

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        kill_switch: KillSwitch | None = None,
        portfolio_service: PaperPortfolioService | None = None,
        database: OperationalDatabase | None = None,
    ) -> None:
        """Create the risk manager and guard dependencies."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        limits = self.config.risk_limits

        # Use provided KillSwitch or create one with persistence
        if kill_switch is not None:
            self.kill_switch = kill_switch
        elif database is not None:
            self.kill_switch = KillSwitch(database=database)
        else:
            self.kill_switch = KillSwitch()

        # Use provided portfolio service or create one
        self.portfolio_service = portfolio_service
        if self.portfolio_service is not None:
            self.portfolio_service.initialize()

        # Derive current equity for guard initialization
        current_equity = self._get_current_equity(limits.capital)

        self.daily_loss_guard = DailyLossGuard(
            capital=current_equity,
            max_daily_loss_percent=limits.max_daily_loss_percent,
            max_daily_profit_lock_percent=limits.max_daily_profit_lock_percent,
        )
        self.trade_loss_guard = TradeLossGuard(
            capital=current_equity,
            max_loss_per_trade_percent=limits.max_loss_per_trade_percent,
        )
        self.position_sizer = PositionSizer(
            capital=current_equity,
            max_loss_per_trade_percent=limits.max_loss_per_trade_percent,
            max_loss_per_trade_absolute=getattr(limits, "max_loss_per_trade_absolute", None),
        )
        self.exposure_limits = ExposureLimits(
            max_trades_per_day=limits.max_trades_per_day,
            max_open_positions=limits.max_open_positions,
            max_consecutive_losses=limits.max_consecutive_losses,
            max_margin_usage_percent=limits.max_margin_usage_percent,
        )
        self.slippage_model = SlippageModel(
            max_bid_ask_spread_percent=limits.max_bid_ask_spread_percent,
            max_slippage_percent=limits.max_slippage_percent,
        )

    def _get_current_equity(self, default_capital: float) -> float:
        """Get current equity from portfolio service or use default."""
        if self.portfolio_service is None:
            return float(default_capital)
        return self.portfolio_service.get_current_equity()

    def validate(
        self,
        plan: StrategyPlan,
        *,
        account_context: Mapping[str, Any] | None = None,
        market_context: Mapping[str, Any] | None = None,
        regime: RegimeResult | RegimeType | str | None = None,
        prediction: PredictionResult | None = None,
        timestamp: datetime | None = None,
    ) -> RiskDecision:
        """Validate all configured risk checks and return a RiskDecision."""

        account = dict(account_context or {})
        market = dict(market_context or {})
        evaluated_at = ensure_ist(timestamp or now_ist())

        # Get authoritative current equity from portfolio service
        equity_source = "portfolio_service"
        if self.portfolio_service is not None:
            current_equity = self.portfolio_service.get_current_equity()
            state = self.portfolio_service.get_state()
            # Override account context with authoritative state
            account["current_equity"] = current_equity
            account["realized_pnl"] = state.get("realized_pnl", 0.0)
            account["unrealized_pnl"] = state.get("unrealized_pnl", 0.0)
            account["daily_realized_pnl"] = state.get("daily_realized_pnl", 0.0)
            account["daily_unrealized_pnl"] = state.get("daily_unrealized_pnl", 0.0)
            account["pending_order_risk"] = state.get("pending_order_risk", 0.0)
            account["partial_fill_risk"] = state.get("partial_fill_risk", 0.0)
            account["pending_order_margin"] = state.get("margin_reserved", 0.0)
            account["partial_fill_margin"] = state.get("margin_reserved", 0.0)
            account["margin_used"] = state.get("margin_used", 0.0)
            account["consecutive_losses"] = state.get("consecutive_losses", 0)
            account["open_positions"] = state.get("open_strategy_count", 0)
            account["trades_today"] = state.get("trade_count", 0)
        else:
            current_equity = _optional_float(account.get("current_equity"))
            equity_source = "missing"
            if current_equity is None:
                initial_capital = _optional_float(account.get("initial_capital"))
                realized_pnl = _optional_float(account.get("realized_pnl"))
                unrealized_pnl = _optional_float(account.get("unrealized_pnl"))
                accrued_costs = _optional_float(account.get("accrued_costs"))
                if initial_capital is not None:
                    current_equity = initial_capital + (realized_pnl or 0.0) + (unrealized_pnl or 0.0) - (accrued_costs or 0.0)
                    equity_source = "derived"
                else:
                    return self._reject(
                        "Missing authoritative account state.",
                        checks={"positive_current_equity": False, "authoritative_account_state": False},
                        warnings=[],
                        timestamp=evaluated_at,
                        metadata={
                            "current_equity_source": equity_source,
                            "derived_current_equity": None,
                            "percentage_risk_limit": self.config.risk_limits.max_loss_per_trade_percent,
                            "absolute_cap": getattr(self.config.risk_limits, "max_loss_per_trade_absolute", None),
                            "final_max_risk_amount": 0.0,
                            "remaining_risk": 0.0,
                            "reserved_risk": float(account.get("pending_order_risk", 0.0)) + float(account.get("partial_fill_risk", 0.0)),
                        },
                    )
            else:
                equity_source = "explicit"
            account["current_equity"] = current_equity

        accounting = PortfolioAccounting(
            initial_capital=float(account.get("initial_capital", self.config.risk_limits.capital)),
            realized_pnl=float(account.get("realized_pnl", 0.0)),
            unrealized_pnl=float(account.get("unrealized_pnl", 0.0)),
            accrued_costs=float(account.get("accrued_costs", 0.0)),
            pending_order_risk=float(account.get("pending_order_risk", 0.0)),
            partial_fill_risk=float(account.get("partial_fill_risk", 0.0)),
            pending_order_margin=float(account.get("pending_order_margin", 0.0)),
            partial_fill_margin=float(account.get("partial_fill_margin", 0.0)),
        )

        if current_equity <= 0:
            return self._reject(
                "Current equity is zero or negative.",
                checks={"positive_current_equity": False},
                warnings=[],
                timestamp=evaluated_at,
                metadata={
                    "current_equity": current_equity,
                    "current_equity_source": equity_source,
                    "derived_current_equity": current_equity,
                    "percentage_risk_limit": self.config.risk_limits.max_loss_per_trade_percent,
                    "absolute_cap": getattr(self.config.risk_limits, "max_loss_per_trade_absolute", None),
                    "final_max_risk_amount": 0.0,
                    "remaining_risk": 0.0,
                    "reserved_risk": float(account.get("pending_order_risk", 0.0)) + float(account.get("partial_fill_risk", 0.0)),
                },
            )

        checks: dict[str, bool] = {}
        warnings: list[str] = []

        if self.kill_switch.active:
            state = self.kill_switch.state()
            return self._reject(
                "Kill switch is active.",
                checks={"kill_switch_inactive": False},
                warnings=warnings,
                timestamp=evaluated_at,
                metadata={"kill_switch": state.__dict__, **self._equity_metadata(current_equity, equity_source, accounting.reserved_risk)},
            )
        checks["kill_switch_inactive"] = True

        if not is_market_open(
            evaluated_at,
            market_start=self.config.settings.market_start_time,
            market_end=self.config.settings.market_end_time,
        ):
            return self._reject("Market is closed.", checks=checks | {"market_hours": False}, warnings=warnings, timestamp=evaluated_at, metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk))
        checks["market_hours"] = True

        if not can_place_new_trade(
            evaluated_at,
            market_start=self.config.settings.market_start_time,
            no_new_trade_after=self.config.settings.no_new_trade_after_time,
        ):
            return self._reject(
                "No new trade allowed after configured cutoff.",
                checks=checks | {"no_new_trade_after": False},
                warnings=warnings,
                timestamp=evaluated_at,
                metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk),
            )
        checks["no_new_trade_after"] = True

        if self._is_event_risk(regime, prediction):
            return self._reject("Event-risk regime blocks new trades.", checks=checks | {"event_risk_regime": False}, warnings=warnings, timestamp=evaluated_at, metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk))
        checks["event_risk_regime"] = True
        news_risk_level = str(market.get("event_risk_level") or plan.metadata.get("event_risk_level") or "").upper()
        if news_risk_level in {"HIGH", "EXTREME"}:
            return self._reject(
                f"News event risk {news_risk_level} blocks new trades.",
                checks=checks | {"news_event_risk": False},
                warnings=warnings,
                timestamp=evaluated_at,
                metadata={"news_event_risk": dict(market.get("news_event_risk", {})) if isinstance(market.get("news_event_risk"), Mapping) else {}, **self._equity_metadata(current_equity, equity_source, accounting.reserved_risk)},
            )
        checks["news_event_risk"] = True

        ev_check = self._expected_value_check(plan)
        checks["positive_expected_value"] = ev_check is None
        if ev_check is not None:
            return self._reject(ev_check, checks=checks, warnings=warnings, timestamp=evaluated_at, metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk))

        trade_loss = TradeLossGuard(
            capital=current_equity,
            max_loss_per_trade_percent=self.config.risk_limits.max_loss_per_trade_percent,
        ).check(plan)
        checks["strategy_max_loss_defined"] = plan.max_loss is not None and plan.max_loss > 0
        checks["per_trade_max_loss"] = trade_loss.approved
        if not trade_loss.approved:
            return self._reject(
                trade_loss.reason,
                checks=checks,
                warnings=warnings,
                timestamp=evaluated_at,
                max_risk_amount=trade_loss.max_risk_amount,
                risk_used=trade_loss.strategy_max_loss,
                metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk),
            )

        daily = self.daily_loss_guard.check(
            daily_pnl=float(account.get("daily_pnl", 0.0)),
            peak_daily_pnl=_optional_float(account.get("peak_daily_pnl")),
        )
        checks["daily_max_loss"] = daily.approved and "loss" not in daily.reason.lower()
        checks["daily_profit_lock"] = daily.approved and "profit lock triggered" not in daily.reason.lower()
        warnings.extend(daily.warnings)
        if not daily.approved:
            return self._reject(daily.reason, checks=checks, warnings=warnings, timestamp=evaluated_at, metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk))

        sizing = self.position_sizer.size(
            plan,
            event_risk_level=news_risk_level or None,
            position_size_multiplier=_optional_float(market.get("position_size_multiplier")),
            current_equity=current_equity,
            reserved_risk=accounting.reserved_risk,
        )
        checks["position_sizing"] = sizing.approved
        if not sizing.approved:
            return self._reject(
                sizing.reason,
                checks=checks,
                warnings=warnings,
                timestamp=evaluated_at,
                allowed_quantity=sizing.allowed_quantity,
                max_risk_amount=sizing.max_risk_amount,
                risk_used=sizing.plan_risk,
                metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk, sizing=sizing),
            )

        exposure = self.exposure_limits.check(
            trades_today=int(account.get("trades_today", 0)),
            open_positions=int(account.get("open_positions", 0)),
            consecutive_losses=int(account.get("consecutive_losses", 0)),
            margin_used=float(account.get("margin_used", 0.0)),
            capital=current_equity,
            additional_margin=float(plan.metadata.get("margin_estimate_placeholder", plan.max_loss or 0.0)) + accounting.reserved_margin,
        )
        checks["max_trades_per_day"] = exposure.approved or exposure.reason != "Max trades per day reached."
        checks["max_open_positions"] = exposure.approved or exposure.reason != "Max open positions reached."
        checks["max_consecutive_losses"] = exposure.approved or exposure.reason != "Max consecutive losses reached."
        checks["max_margin_usage"] = exposure.approved or exposure.reason != "Max margin usage exceeded."
        warnings.extend(exposure.warnings)
        if not exposure.approved:
            return self._reject(exposure.reason, checks=checks, warnings=warnings, timestamp=evaluated_at, metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk))

        slippage = self.slippage_model.check(plan, market_context=market)
        checks["bid_ask_spread_limit"] = slippage.approved or slippage.reason != "Bid-ask spread limit exceeded."
        checks["slippage_limit"] = slippage.approved or slippage.reason != "Slippage limit exceeded."
        warnings.extend(slippage.warnings)
        if not slippage.approved:
            return self._reject(slippage.reason, checks=checks, warnings=warnings, timestamp=evaluated_at, metadata=self._equity_metadata(current_equity, equity_source, accounting.reserved_risk))

        checks["daily_max_loss"] = True
        checks["daily_profit_lock"] = True
        checks["max_trades_per_day"] = True
        checks["max_open_positions"] = True
        checks["max_consecutive_losses"] = True
        checks["max_margin_usage"] = True
        checks["bid_ask_spread_limit"] = True
        checks["slippage_limit"] = True

        decision = RiskDecision(
            approved=True,
            timestamp=evaluated_at,
            reason="Risk approved.",
            rejection_reason=None,
            max_allowed_quantity=sizing.allowed_quantity,
            allowed_quantity=sizing.allowed_quantity,
            capital_at_risk=sizing.plan_risk,
            max_risk_amount=sizing.max_risk_amount,
            risk_used=sizing.plan_risk,
            margin_required=float(plan.metadata.get("margin_estimate_placeholder", plan.max_loss or 0.0)),
            checks=checks,
            warnings=tuple(warnings),
            metadata={
                "daily_pnl": daily.daily_pnl,
                "margin_usage_percent": exposure.margin_usage_percent,
                "bid_ask_spread_percent": slippage.max_spread_percent,
                "slippage_percent": slippage.estimated_slippage_percent,
                "current_equity": current_equity,
                "current_equity_source": equity_source,
                "derived_current_equity": current_equity,
                "percentage_risk_limit": self.config.risk_limits.max_loss_per_trade_percent,
                "absolute_cap": getattr(self.config.risk_limits, "max_loss_per_trade_absolute", None),
                "final_max_risk_amount": sizing.max_risk_amount,
                "remaining_risk": sizing.max_risk_amount,
                "reserved_risk": accounting.reserved_risk,
                "reserved_margin": accounting.reserved_margin,
            },
        )
        LOGGER.info("Risk approved", extra={"strategy_id": plan.strategy_id, "allowed_quantity": sizing.allowed_quantity})
        return decision

    def trigger_kill_switch(self, reason: str) -> None:
        """Activate the risk kill switch."""

        self.kill_switch.trigger(reason)
        LOGGER.critical("Risk kill switch triggered", extra={"reason": reason})

    def reset_kill_switch(self) -> None:
        """Reset the risk kill switch."""

        self.kill_switch.reset()
        LOGGER.warning("Risk kill switch reset")

    def _equity_metadata(
        self,
        current_equity: float,
        equity_source: str,
        reserved_risk: float = 0.0,
        sizing: PositionSizingResult | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "current_equity": current_equity,
            "current_equity_source": equity_source,
            "derived_current_equity": current_equity,
            "percentage_risk_limit": self.config.risk_limits.max_loss_per_trade_percent,
            "absolute_cap": getattr(self.config.risk_limits, "max_loss_per_trade_absolute", None),
            "reserved_risk": reserved_risk,
        }
        if sizing is not None:
            meta["final_max_risk_amount"] = sizing.max_risk_amount
            meta["remaining_risk"] = sizing.max_risk_amount
        return meta

    def _reject(
        self,
        reason: str,
        *,
        checks: dict[str, bool],
        warnings: list[str],
        timestamp: datetime,
        allowed_quantity: int | None = 0,
        max_risk_amount: float = 0.0,
        risk_used: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> RiskDecision:
        LOGGER.warning("Risk rejected", extra={"reason": reason})
        return RiskDecision(
            approved=False,
            timestamp=timestamp,
            reason=reason,
            rejection_reason=reason,
            max_allowed_quantity=allowed_quantity,
            allowed_quantity=allowed_quantity,
            capital_at_risk=risk_used,
            max_risk_amount=max_risk_amount,
            risk_used=risk_used,
            margin_required=0.0,
            checks=checks,
            warnings=tuple(warnings),
            metadata=metadata or {},
        )

    @staticmethod
    def _is_event_risk(regime: RegimeResult | RegimeType | str | None, prediction: PredictionResult | None) -> bool:
        if prediction and not prediction.trade_allowed and "event" in prediction.reason.lower():
            return True
        if regime is None:
            return False
        if isinstance(regime, RegimeResult):
            name = (regime.regime_name or regime.regime.value).lower()
            return regime.event_risk_score >= 0.7 or "event" in name or not regime.allowed_to_trade and "event" in name
        if isinstance(regime, RegimeType):
            return False
        return "event" in str(regime).lower()

    @staticmethod
    def _expected_value_check(plan: StrategyPlan) -> str | None:
        expected_value = _optional_float(plan.metadata.get("expected_value"))
        if expected_value is not None and expected_value < 0:
            return "Strategy expected value is negative."
        probability = _optional_float(plan.metadata.get("probability_of_profit"))
        if probability is not None and not 0.0 <= probability <= 1.0:
            return "Strategy probability of profit is invalid."
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None