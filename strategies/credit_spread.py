"""Defined-risk credit spread advisory strategies."""

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

CreditSpreadVariant = Literal["bull_put", "bear_call"]


@dataclass
class CreditSpreadStrategy(BaseStrategy):
    """Bull put or bear call spread."""

    name: str = "Credit Spread"
    strategy_type: StrategyType = StrategyType.CREDIT_SPREAD
    allowed_regimes: tuple[str, ...] = ("trending", "range_bound", "low_volatility")
    forbidden_regimes: tuple[str, ...] = ()
    required_confidence: float = 0.65
    variant: CreditSpreadVariant | None = None
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
        """Build a credit spread plan."""

        reason = missing_chain_plan_reason(option_chain_snapshot)
        if reason:
            return _blocked_plan(self, prediction, reason)
        assert option_chain_snapshot is not None
        variant = self.variant or ("bull_put" if prediction.bullish_probability >= prediction.bearish_probability else "bear_call")
        short_contract, long_contract = self._contracts(option_chain_snapshot, variant)
        if short_contract is None or long_contract is None:
            return _blocked_plan(self, prediction, f"Insufficient strikes for {variant} credit spread.")

        quantity = lot_size * max(1, max_lots)
        legs = (
            make_leg(short_contract, OrderSide.SELL, quantity),
            make_leg(long_contract, OrderSide.BUY, quantity),
        )
        credit = max(net_premium(legs), 0.0)
        width = strike_width(short_contract, long_contract)
        max_profit = credit * quantity
        max_loss = max(width - credit, 0.0) * quantity
        breakeven = (
            short_contract.strike - credit if variant == "bull_put" else short_contract.strike + credit,
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
            rationale=f"{variant} credit spread selected for directional premium collection.",
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
                    "Exit at 50% of credit captured.",
                    "Exit if loss reaches configured stop.",
                    "Exit if spot closes beyond short strike.",
                    "Exit before expiry if spread liquidity weakens.",
                ),
                "margin_estimate_placeholder": metrics.margin_estimate_placeholder,
                "metrics": metrics.as_dict(),
                "advisory_only": True,
            },
        )

    def _contracts(
        self,
        snapshot: OptionChainSnapshot,
        variant: CreditSpreadVariant,
    ) -> tuple[OptionContract | None, OptionContract | None]:
        if variant == "bull_put":
            short_put = nearest_otm(snapshot, "PE", self.short_steps)
            return short_put, farther_otm(snapshot, short_put, self.hedge_steps) if short_put else None
        short_call = nearest_otm(snapshot, "CE", self.short_steps)
        return short_call, farther_otm(snapshot, short_call, self.hedge_steps) if short_call else None


def _entry_reason(variant: CreditSpreadVariant) -> str:
    if variant == "bull_put":
        return "Bullish or neutral view; sell OTM put and buy farther OTM put for defined risk."
    return "Bearish or neutral view; sell OTM call and buy farther OTM call for defined risk."


def _blocked_plan(strategy: CreditSpreadStrategy, prediction: PredictionResult, reason: str) -> StrategyPlan:
    return StrategyPlan(
        strategy_id=f"credit-spread-blocked-{now_ist().strftime('%Y%m%d%H%M%S')}",
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
