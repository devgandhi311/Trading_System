"""Central risk manager for strategy approval."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Mapping

from models import PredictionResult, RegimeResult, RegimeType, RiskDecision, StrategyPlan
from risk.daily_loss_guard import DailyLossGuard
from risk.exposure_limits import ExposureLimits
from risk.kill_switch import KillSwitch
from risk.position_sizing import PositionSizer
from risk.slippage_model import SlippageModel
from risk.trade_loss_guard import TradeLossGuard
from utils.config_loader import ConfigLoader
from utils.time_utils import can_place_new_trade, ensure_ist, is_market_open, now_ist

LOGGER = logging.getLogger(__name__)


class RiskManager:
    """Validate strategy plans against configured risk limits."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        """Create the risk manager and guard dependencies."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        limits = self.config.risk_limits
        self.kill_switch = kill_switch or KillSwitch()
        self.daily_loss_guard = DailyLossGuard(
            capital=limits.capital,
            max_daily_loss_percent=limits.max_daily_loss_percent,
            max_daily_profit_lock_percent=limits.max_daily_profit_lock_percent,
        )
        self.trade_loss_guard = TradeLossGuard(
            capital=limits.capital,
            max_loss_per_trade_percent=limits.max_loss_per_trade_percent,
        )
        self.position_sizer = PositionSizer(
            capital=limits.capital,
            max_loss_per_trade_percent=limits.max_loss_per_trade_percent,
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
        checks: dict[str, bool] = {}
        warnings: list[str] = []

        if self.kill_switch.active:
            state = self.kill_switch.state()
            return self._reject(
                "Kill switch is active.",
                checks={"kill_switch_inactive": False},
                warnings=warnings,
                timestamp=evaluated_at,
                metadata={"kill_switch": state.__dict__},
            )
        checks["kill_switch_inactive"] = True

        if not is_market_open(
            evaluated_at,
            market_start=self.config.settings.market_start_time,
            market_end=self.config.settings.market_end_time,
        ):
            return self._reject("Market is closed.", checks=checks | {"market_hours": False}, warnings=warnings, timestamp=evaluated_at)
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
            )
        checks["no_new_trade_after"] = True

        if self._is_event_risk(regime, prediction):
            return self._reject("Event-risk regime blocks new trades.", checks=checks | {"event_risk_regime": False}, warnings=warnings, timestamp=evaluated_at)
        checks["event_risk_regime"] = True
        news_risk_level = str(market.get("event_risk_level") or plan.metadata.get("event_risk_level") or "").upper()
        if news_risk_level in {"HIGH", "EXTREME"}:
            return self._reject(
                f"News event risk {news_risk_level} blocks new trades.",
                checks=checks | {"news_event_risk": False},
                warnings=warnings,
                timestamp=evaluated_at,
                metadata={"news_event_risk": dict(market.get("news_event_risk", {})) if isinstance(market.get("news_event_risk"), Mapping) else {}},
            )
        checks["news_event_risk"] = True

        ev_check = self._expected_value_check(plan)
        checks["positive_expected_value"] = ev_check is None
        if ev_check is not None:
            return self._reject(ev_check, checks=checks, warnings=warnings, timestamp=evaluated_at)

        trade_loss = self.trade_loss_guard.check(plan)
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
            )

        daily = self.daily_loss_guard.check(
            daily_pnl=float(account.get("daily_pnl", 0.0)),
            peak_daily_pnl=_optional_float(account.get("peak_daily_pnl")),
        )
        checks["daily_max_loss"] = daily.approved and "loss" not in daily.reason.lower()
        checks["daily_profit_lock"] = daily.approved and "profit lock triggered" not in daily.reason.lower()
        warnings.extend(daily.warnings)
        if not daily.approved:
            return self._reject(daily.reason, checks=checks, warnings=warnings, timestamp=evaluated_at)

        sizing = self.position_sizer.size(
            plan,
            event_risk_level=news_risk_level or None,
            position_size_multiplier=_optional_float(market.get("position_size_multiplier")),
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
            )

        exposure = self.exposure_limits.check(
            trades_today=int(account.get("trades_today", 0)),
            open_positions=int(account.get("open_positions", 0)),
            consecutive_losses=int(account.get("consecutive_losses", 0)),
            margin_used=float(account.get("margin_used", 0.0)),
            capital=self.config.risk_limits.capital,
            additional_margin=float(plan.metadata.get("margin_estimate_placeholder", plan.max_loss or 0.0)),
        )
        checks["max_trades_per_day"] = exposure.approved or exposure.reason != "Max trades per day reached."
        checks["max_open_positions"] = exposure.approved or exposure.reason != "Max open positions reached."
        checks["max_consecutive_losses"] = exposure.approved or exposure.reason != "Max consecutive losses reached."
        checks["max_margin_usage"] = exposure.approved or exposure.reason != "Max margin usage exceeded."
        warnings.extend(exposure.warnings)
        if not exposure.approved:
            return self._reject(exposure.reason, checks=checks, warnings=warnings, timestamp=evaluated_at)

        slippage = self.slippage_model.check(plan, market_context=market)
        checks["bid_ask_spread_limit"] = slippage.approved or slippage.reason != "Bid-ask spread limit exceeded."
        checks["slippage_limit"] = slippage.approved or slippage.reason != "Slippage limit exceeded."
        warnings.extend(slippage.warnings)
        if not slippage.approved:
            return self._reject(slippage.reason, checks=checks, warnings=warnings, timestamp=evaluated_at)

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
