"""Iron Condor advisory strategy."""

from __future__ import annotations

from dataclasses import dataclass

from models import OrderSide, PredictionResult, StrategyPlan, StrategyType
from models.market_models import OptionChainSnapshot
from strategies.base_strategy import BaseStrategy
from strategies.option_strategy_utils import (
    StrategyMetrics,
    farther_otm,
    make_leg,
    missing_chain_plan_reason,
    nearest_otm,
    net_premium,
    strike_width,
)
from utils.time_utils import now_ist


@dataclass
class IronCondorStrategy(BaseStrategy):
    """Sell OTM call/put and buy farther OTM call/put hedges."""

    name: str = "Iron Condor"
    strategy_type: StrategyType = StrategyType.IRON_CONDOR
    allowed_regimes: tuple[str, ...] = ("range_bound", "low_volatility")
    forbidden_regimes: tuple[str, ...] = ("high_volatility",)
    required_confidence: float = 0.65
    short_steps: int = 1
    hedge_steps: int = 1

    def build_plan(
        self,
        prediction: PredictionResult,
        option_chain_snapshot: OptionChainSnapshot | None,
        *,
        lot_size: int,
        max_lots: int,
        liquidity_score: float,
    ) -> StrategyPlan:
        """Build an iron condor plan."""

        reason = missing_chain_plan_reason(option_chain_snapshot)
        if reason:
            return _blocked_plan(self, prediction, reason)
        assert option_chain_snapshot is not None
        quantity = lot_size * max(1, max_lots)
        short_call = nearest_otm(option_chain_snapshot, "CE", self.short_steps)
        short_put = nearest_otm(option_chain_snapshot, "PE", self.short_steps)
        long_call = farther_otm(option_chain_snapshot, short_call, self.hedge_steps) if short_call else None
        long_put = farther_otm(option_chain_snapshot, short_put, self.hedge_steps) if short_put else None
        if not all((short_call, long_call, short_put, long_put)):
            return _blocked_plan(self, prediction, "Insufficient OTM strikes for iron condor.")

        legs = (
            make_leg(short_call, OrderSide.SELL, quantity),
            make_leg(long_call, OrderSide.BUY, quantity),
            make_leg(short_put, OrderSide.SELL, quantity),
            make_leg(long_put, OrderSide.BUY, quantity),
        )
        credit = max(net_premium(legs), 0.0)
        call_width = strike_width(short_call, long_call)
        put_width = strike_width(short_put, long_put)
        width = max(call_width, put_width)
        max_profit = credit * quantity
        max_loss = max(width - credit, 0.0) * quantity
        breakeven = (short_put.strike - credit, short_call.strike + credit)
        metrics = StrategyMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            target_profit=max_profit * 0.5,
            stop_loss=max_loss * 0.5,
            margin_estimate_placeholder=max_loss,
        )
        valid, validation_reason = self.validate(prediction, liquidity_score=liquidity_score, max_loss=max_loss)
        return StrategyPlan(
            strategy_id=f"iron-condor-{now_ist().strftime('%Y%m%d%H%M%S')}",
            strategy_type=self.strategy_type,
            symbol=prediction.underlying or prediction.symbol,
            created_at=now_ist(),
            legs=legs if valid else (),
            confidence=prediction.confidence_score,
            rationale="Range-bound premium collection with defined-risk wings.",
            max_loss=max_loss,
            expected_profit=max_profit,
            metadata={
                "valid": valid,
                "validation_reason": validation_reason,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven": breakeven,
                "target_profit": metrics.target_profit,
                "stop_loss": metrics.stop_loss,
                "entry_reason": "Sideways or low-volatility regime with price expected to remain inside short strikes.",
                "exit_conditions": (
                    "Exit at 50% of credit captured.",
                    "Exit if loss reaches configured stop.",
                    "Exit before expiry-day liquidity deteriorates.",
                    "Exit if spot breaches either short strike.",
                ),
                "margin_estimate_placeholder": metrics.margin_estimate_placeholder,
                "metrics": metrics.as_dict(),
                "advisory_only": True,
            },
        )


def _blocked_plan(strategy: IronCondorStrategy, prediction: PredictionResult, reason: str) -> StrategyPlan:
    return StrategyPlan(
        strategy_id=f"iron-condor-blocked-{now_ist().strftime('%Y%m%d%H%M%S')}",
        strategy_type=strategy.strategy_type,
        symbol=prediction.underlying or prediction.symbol,
        created_at=now_ist(),
        legs=(),
        confidence=prediction.confidence_score,
        rationale=reason,
        max_loss=0.0,
        expected_profit=0.0,
        metadata={"valid": False, "validation_reason": reason, "advisory_only": True},
    )
