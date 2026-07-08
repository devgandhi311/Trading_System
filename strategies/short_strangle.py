"""Short strangle advisory strategy with strict default disablement."""

from __future__ import annotations

from dataclasses import dataclass

from models import OrderSide, PredictionResult, StrategyPlan, StrategyType
from models.market_models import OptionChainSnapshot, OptionContract
from strategies.base_strategy import BaseStrategy
from strategies.option_strategy_utils import (
    StrategyMetrics,
    farther_otm,
    make_leg,
    missing_chain_plan_reason,
    nearest_otm,
    net_premium,
)
from utils.time_utils import now_ist


@dataclass
class ShortStrangleStrategy(BaseStrategy):
    """Short strangle, disabled by default and requiring emergency hedge logic."""

    name: str = "Short Strangle"
    strategy_type: StrategyType = StrategyType.SHORT_STRANGLE
    allowed_regimes: tuple[str, ...] = ("range_bound", "low_volatility")
    forbidden_regimes: tuple[str, ...] = ("high_volatility", "trending")
    required_confidence: float = 0.75
    enabled: bool = False
    emergency_hedge_enabled: bool = True
    short_steps: int = 1
    hedge_steps: int = 2
    max_loss_multiple_of_credit: float = 2.0

    def build_plan(
        self,
        prediction: PredictionResult,
        option_chain_snapshot: OptionChainSnapshot | None,
        *,
        lot_size: int,
        max_lots: int,
        liquidity_score: float,
    ) -> StrategyPlan:
        """Build a short strangle plan only when explicitly enabled."""

        if not self.enabled:
            return _blocked_plan(self, prediction, "Short strangle is disabled by default.")
        if not self.emergency_hedge_enabled:
            return _blocked_plan(self, prediction, "Short strangle requires emergency hedge logic.")
        reason = missing_chain_plan_reason(option_chain_snapshot)
        if reason:
            return _blocked_plan(self, prediction, reason)
        assert option_chain_snapshot is not None
        short_call = nearest_otm(option_chain_snapshot, "CE", self.short_steps)
        short_put = nearest_otm(option_chain_snapshot, "PE", self.short_steps)
        if short_call is None or short_put is None:
            return _blocked_plan(self, prediction, "Insufficient OTM strikes for short strangle.")

        quantity = lot_size * max(1, max_lots)
        legs = (
            make_leg(short_call, OrderSide.SELL, quantity),
            make_leg(short_put, OrderSide.SELL, quantity),
        )
        hedge_legs = self.emergency_hedge_legs(option_chain_snapshot, short_call, short_put, quantity)
        credit = max(net_premium(legs), 0.0)
        max_profit = credit * quantity
        max_loss = max_profit * self.max_loss_multiple_of_credit
        breakeven = (short_put.strike - credit, short_call.strike + credit)
        metrics = StrategyMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            target_profit=max_profit * 0.4,
            stop_loss=max_loss,
            margin_estimate_placeholder=max_loss * 1.25,
        )
        valid, validation_reason = self.validate(prediction, liquidity_score=liquidity_score, max_loss=max_loss)
        if prediction.breakout_failure_probability > 0.4:
            valid = False
            validation_reason = "Breakout failure and liquidity-trap risk too high for naked premium selling."
        return StrategyPlan(
            strategy_id=f"short-strangle-{now_ist().strftime('%Y%m%d%H%M%S')}",
            strategy_type=self.strategy_type,
            symbol=prediction.underlying or prediction.symbol,
            created_at=now_ist(),
            legs=legs if valid else (),
            confidence=prediction.confidence_score,
            rationale="Strict-risk short strangle premium collection; advisory only.",
            max_loss=max_loss,
            expected_profit=max_profit,
            metadata={
                "valid": valid,
                "validation_reason": validation_reason,
                "enabled": self.enabled,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven": breakeven,
                "target_profit": metrics.target_profit,
                "stop_loss": metrics.stop_loss,
                "entry_reason": "Low-volatility, range-bound regime with strict loss cap and hedge readiness.",
                "exit_conditions": (
                    "Exit at 40% of credit captured.",
                    "Exit immediately at stop loss.",
                    "Emergency hedge if spot breaches either short strike.",
                    "Exit if volatility expands or liquidity score deteriorates.",
                ),
                "emergency_hedge_logic": {
                    "enabled": self.emergency_hedge_enabled,
                    "trigger": "Spot breaches short strike or loss reaches 50% of max loss.",
                    "hedge_legs": [leg.metadata | {"symbol": leg.symbol, "side": leg.side.value} for leg in hedge_legs],
                },
                "margin_estimate_placeholder": metrics.margin_estimate_placeholder,
                "metrics": metrics.as_dict(),
                "advisory_only": True,
            },
        )

    def emergency_hedge_legs(
        self,
        snapshot: OptionChainSnapshot,
        short_call: OptionContract,
        short_put: OptionContract,
        quantity: int,
    ) -> tuple:
        """Return farther OTM buy hedges for emergency conversion to defined risk."""

        hedge_call = farther_otm(snapshot, short_call, self.hedge_steps)
        hedge_put = farther_otm(snapshot, short_put, self.hedge_steps)
        legs = []
        if hedge_call is not None:
            legs.append(make_leg(hedge_call, OrderSide.BUY, quantity))
        if hedge_put is not None:
            legs.append(make_leg(hedge_put, OrderSide.BUY, quantity))
        return tuple(legs)


def _blocked_plan(strategy: ShortStrangleStrategy, prediction: PredictionResult, reason: str) -> StrategyPlan:
    return StrategyPlan(
        strategy_id=f"short-strangle-blocked-{now_ist().strftime('%Y%m%d%H%M%S')}",
        strategy_type=strategy.strategy_type,
        symbol=prediction.underlying or prediction.symbol,
        created_at=now_ist(),
        legs=(),
        confidence=prediction.confidence_score,
        rationale=reason,
        max_loss=0.0,
        expected_profit=0.0,
        metadata={
            "valid": False,
            "enabled": strategy.enabled,
            "validation_reason": reason,
            "emergency_hedge_logic": {"enabled": strategy.emergency_hedge_enabled},
            "advisory_only": True,
        },
    )
