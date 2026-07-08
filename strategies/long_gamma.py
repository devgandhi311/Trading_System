"""Long gamma advisory strategies: long straddle and long strangle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from models import OrderSide, PredictionResult, StrategyPlan, StrategyType
from models.market_models import OptionChainSnapshot, OptionContract
from strategies.base_strategy import BaseStrategy
from strategies.option_strategy_utils import (
    StrategyMetrics,
    atm_contract,
    make_leg,
    missing_chain_plan_reason,
    nearest_otm,
    net_premium,
)
from utils.time_utils import now_ist

LongGammaVariant = Literal["long_straddle", "long_strangle"]


@dataclass
class LongGammaStrategy(BaseStrategy):
    """Buy call and put exposure for large moves."""

    name: str = "Long Gamma"
    strategy_type: StrategyType = StrategyType.LONG_GAMMA
    allowed_regimes: tuple[str, ...] = ("high_volatility", "trending")
    forbidden_regimes: tuple[str, ...] = ()
    required_confidence: float = 0.65
    variant: LongGammaVariant = "long_straddle"
    otm_steps: int = 1

    def build_plan(
        self,
        prediction: PredictionResult,
        option_chain_snapshot: OptionChainSnapshot | None,
        *,
        lot_size: int,
        max_lots: int,
        liquidity_score: float,
    ) -> StrategyPlan:
        """Build long straddle or long strangle plan."""

        reason = missing_chain_plan_reason(option_chain_snapshot)
        if reason:
            return _blocked_plan(self, prediction, reason)
        assert option_chain_snapshot is not None
        call, put = self._contracts(option_chain_snapshot)
        if call is None or put is None:
            return _blocked_plan(self, prediction, f"Insufficient strikes for {self.variant}.")

        quantity = lot_size * max(1, max_lots)
        legs = (
            make_leg(call, OrderSide.BUY, quantity),
            make_leg(put, OrderSide.BUY, quantity),
        )
        debit = abs(net_premium(legs))
        max_loss = debit * quantity
        max_profit = float("inf")
        breakeven = (put.strike - debit, call.strike + debit)
        metrics = StrategyMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            target_profit=max_loss * 0.6,
            stop_loss=max_loss * 0.5,
            margin_estimate_placeholder=max_loss,
        )
        valid, validation_reason = self.validate(prediction, liquidity_score=liquidity_score, max_loss=max_loss)
        return StrategyPlan(
            strategy_id=f"{self.variant}-{now_ist().strftime('%Y%m%d%H%M%S')}",
            strategy_type=self.strategy_type,
            symbol=prediction.underlying or prediction.symbol,
            created_at=now_ist(),
            legs=legs if valid else (),
            confidence=prediction.confidence_score,
            rationale=f"{self.variant} selected for expected expansion in realized movement.",
            max_loss=max_loss,
            expected_profit=metrics.target_profit,
            metadata={
                "valid": valid,
                "validation_reason": validation_reason,
                "variant": self.variant,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven": breakeven,
                "target_profit": metrics.target_profit,
                "stop_loss": metrics.stop_loss,
                "entry_reason": "Expected move or event risk justifies owning convexity instead of selling premium.",
                "exit_conditions": (
                    "Exit at target profit.",
                    "Exit if premium loses 50%.",
                    "Exit after event passes if implied volatility collapses.",
                    "Exit near expiry if gamma risk is no longer desired.",
                ),
                "margin_estimate_placeholder": metrics.margin_estimate_placeholder,
                "metrics": metrics.as_dict(),
                "advisory_only": True,
            },
        )

    def _contracts(self, snapshot: OptionChainSnapshot) -> tuple[OptionContract | None, OptionContract | None]:
        if self.variant == "long_strangle":
            return nearest_otm(snapshot, "CE", self.otm_steps), nearest_otm(snapshot, "PE", self.otm_steps)
        return atm_contract(snapshot, "CE"), atm_contract(snapshot, "PE")


def _blocked_plan(strategy: LongGammaStrategy, prediction: PredictionResult, reason: str) -> StrategyPlan:
    return StrategyPlan(
        strategy_id=f"long-gamma-blocked-{now_ist().strftime('%Y%m%d%H%M%S')}",
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
