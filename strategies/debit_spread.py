"""Defined-risk debit spread advisory strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    strike_width,
)
from utils.time_utils import now_ist

DebitSpreadVariant = Literal["bull_call", "bear_put"]


@dataclass
class DebitSpreadStrategy(BaseStrategy):
    """Bull call or bear put spread."""

    name: str = "Debit Spread"
    strategy_type: StrategyType = StrategyType.DEBIT_SPREAD
    allowed_regimes: tuple[str, ...] = ("trending", "high_volatility")
    forbidden_regimes: tuple[str, ...] = ()
    required_confidence: float = 0.65
    variant: DebitSpreadVariant | None = None
    long_steps: int = 1
    short_steps: int = 1

    def build_plan(
        self,
        prediction: PredictionResult,
        option_chain_snapshot: OptionChainSnapshot | None,
        *,
        lot_size: int,
        max_lots: int,
        liquidity_score: float,
    ) -> StrategyPlan:
        """Build a debit spread plan."""

        reason = missing_chain_plan_reason(option_chain_snapshot)
        if reason:
            return _blocked_plan(self, prediction, reason)
        assert option_chain_snapshot is not None
        variant = self.variant or ("bull_call" if prediction.bullish_probability >= prediction.bearish_probability else "bear_put")
        long_contract, short_contract = self._contracts(option_chain_snapshot, variant)
        if long_contract is None or short_contract is None:
            return _blocked_plan(self, prediction, f"Insufficient strikes for {variant} debit spread.")

        quantity = lot_size * max(1, max_lots)
        legs = (
            make_leg(long_contract, OrderSide.BUY, quantity),
            make_leg(short_contract, OrderSide.SELL, quantity),
        )
        debit = abs(min(net_premium(legs), 0.0))
        width = strike_width(long_contract, short_contract)
        max_profit = max(width - debit, 0.0) * quantity
        max_loss = debit * quantity
        breakeven = (
            long_contract.strike + debit if variant == "bull_call" else long_contract.strike - debit,
        )
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
            strategy_id=f"{variant}-{now_ist().strftime('%Y%m%d%H%M%S')}",
            strategy_type=self.strategy_type,
            symbol=prediction.underlying or prediction.symbol,
            created_at=now_ist(),
            legs=legs if valid else (),
            confidence=prediction.confidence_score,
            rationale=f"{variant} debit spread selected for directional participation with capped risk.",
            max_loss=max_loss,
            expected_profit=max_profit,
            metadata={
                "valid": valid,
                "validation_reason": validation_reason,
                "variant": variant,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven": breakeven,
                "target_profit": metrics.target_profit,
                "stop_loss": metrics.stop_loss,
                "entry_reason": _entry_reason(variant),
                "exit_conditions": (
                    "Exit at target profit.",
                    "Exit if debit loses 50%.",
                    "Exit if directional thesis invalidates.",
                    "Exit before expiry if liquidity deteriorates.",
                ),
                "margin_estimate_placeholder": metrics.margin_estimate_placeholder,
                "metrics": metrics.as_dict(),
                "advisory_only": True,
            },
        )

    def _contracts(
        self,
        snapshot: OptionChainSnapshot,
        variant: DebitSpreadVariant,
    ) -> tuple[OptionContract | None, OptionContract | None]:
        if variant == "bull_call":
            long_call = nearest_otm(snapshot, "CE", self.long_steps)
            return long_call, farther_otm(snapshot, long_call, self.short_steps) if long_call else None
        long_put = nearest_otm(snapshot, "PE", self.long_steps)
        return long_put, farther_otm(snapshot, long_put, self.short_steps) if long_put else None


def _entry_reason(variant: DebitSpreadVariant) -> str:
    if variant == "bull_call":
        return "Bullish view; buy OTM call and sell farther OTM call to reduce debit."
    return "Bearish view; buy OTM put and sell farther OTM put to reduce debit."


def _blocked_plan(strategy: DebitSpreadStrategy, prediction: PredictionResult, reason: str) -> StrategyPlan:
    return StrategyPlan(
        strategy_id=f"debit-spread-blocked-{now_ist().strftime('%Y%m%d%H%M%S')}",
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
