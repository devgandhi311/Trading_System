"""Base strategy abstractions that produce advisory StrategyPlan objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models import OrderSide, OrderType, PredictionResult, StrategyLeg, StrategyPlan, StrategyType
from models.market_models import OptionChainSnapshot, OptionContract
from utils.math_utils import clamp
from utils.time_utils import now_ist


@dataclass
class BaseStrategy:
    """Base class for strategy plan generation without order placement."""

    name: str
    strategy_type: StrategyType
    allowed_regimes: tuple[str, ...]
    forbidden_regimes: tuple[str, ...] = field(default_factory=tuple)
    required_confidence: float = 0.65

    def generate_legs(
        self,
        prediction: PredictionResult,
        option_chain_snapshot: OptionChainSnapshot | None,
        *,
        lot_size: int,
        max_lots: int,
    ) -> tuple[StrategyLeg, ...]:
        """Generate advisory legs for this strategy type."""

        quantity = max(1, max_lots) * lot_size
        if option_chain_snapshot is None or not option_chain_snapshot.contracts:
            return (
                StrategyLeg(
                    symbol=prediction.underlying or prediction.symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.LIMIT,
                    metadata={"paper_advisory": True, "reason": "No option chain available"},
                ),
            )

        if self.strategy_type == StrategyType.MOMENTUM_BREAKOUT:
            option_type = "CE" if prediction.bullish_probability >= prediction.bearish_probability else "PE"
            contract = self._nearest_atm(option_chain_snapshot, option_type=option_type)
            return (self._leg(contract, OrderSide.BUY, quantity),)

        if self.strategy_type == StrategyType.VOLATILITY_SCALPING:
            call = self._nearest_atm(option_chain_snapshot, option_type="CE")
            put = self._nearest_atm(option_chain_snapshot, option_type="PE")
            return tuple(self._leg(contract, OrderSide.BUY, quantity) for contract in (call, put) if contract)

        if self.strategy_type == StrategyType.MEAN_REVERSION:
            call = self._nearest_otm(option_chain_snapshot, option_type="CE")
            put = self._nearest_otm(option_chain_snapshot, option_type="PE")
            return tuple(self._leg(contract, OrderSide.SELL, quantity) for contract in (call, put) if contract)

        return ()

    def validate(self, prediction: PredictionResult, *, liquidity_score: float, max_loss: float | None) -> tuple[bool, str]:
        """Validate confidence, regime, liquidity, and estimated loss filters."""

        regime = prediction.regime.value
        if self.forbidden_regimes and regime in self.forbidden_regimes:
            return False, f"{self.name} is forbidden in {regime} regime"
        if self.allowed_regimes and regime not in self.allowed_regimes:
            return False, f"{self.name} is not configured for {regime} regime"
        if prediction.confidence_score < self.required_confidence:
            return False, "Prediction confidence is below strategy threshold"
        if liquidity_score < 0.35:
            return False, "Liquidity score is too low"
        if max_loss is not None and max_loss <= 0:
            return False, "Estimated max loss is invalid"
        return True, "Strategy validation passed"

    def estimate_payoff(self, legs: tuple[StrategyLeg, ...], spot_price: float, expected_move_percent: float) -> dict[str, float]:
        """Estimate simple upside/downside payoff around the expected move range."""

        move = spot_price * expected_move_percent / 100.0
        max_profit = self.estimate_max_profit(legs, spot_price, expected_move_percent)
        max_loss = self.estimate_max_loss(legs, spot_price)
        return {
            "downside_spot": spot_price - move,
            "upside_spot": spot_price + move,
            "estimated_max_profit": max_profit if max_profit is not None else 0.0,
            "estimated_max_loss": max_loss if max_loss is not None else 0.0,
        }

    def estimate_max_profit(
        self,
        legs: tuple[StrategyLeg, ...],
        spot_price: float,
        expected_move_percent: float,
    ) -> float | None:
        """Estimate max profit using a conservative heuristic."""

        notional = sum((leg.limit_price or spot_price * 0.01) * leg.quantity for leg in legs)
        if self.strategy_type == StrategyType.MEAN_REVERSION:
            return notional * 0.6
        if self.strategy_type == StrategyType.VOLATILITY_SCALPING:
            return notional * clamp(expected_move_percent, 0.5, 5.0)
        return notional * clamp(expected_move_percent / 2.0, 0.25, 3.0)

    def estimate_max_loss(self, legs: tuple[StrategyLeg, ...], spot_price: float) -> float | None:
        """Estimate max loss for risk filtering."""

        premium = sum((leg.limit_price or spot_price * 0.01) * leg.quantity for leg in legs)
        if not legs:
            return None
        if all(leg.side == OrderSide.BUY for leg in legs):
            return premium
        return premium * 2.0

    def estimate_breakeven(self, legs: tuple[StrategyLeg, ...], spot_price: float) -> tuple[float, ...]:
        """Estimate breakeven levels."""

        premium_points = sum((leg.limit_price or spot_price * 0.01) for leg in legs)
        if self.strategy_type == StrategyType.VOLATILITY_SCALPING:
            return spot_price - premium_points, spot_price + premium_points
        if self.strategy_type == StrategyType.MEAN_REVERSION:
            return spot_price - premium_points / 2.0, spot_price + premium_points / 2.0
        direction = 1 if legs and legs[0].metadata.get("option_type") == "CE" else -1
        return (spot_price + direction * premium_points,)

    def explain_trade(self, prediction: PredictionResult, validation_reason: str) -> str:
        """Explain the strategy plan in terms of prediction context."""

        return (
            f"{self.name} selected with {prediction.confidence_score:.1%} confidence. "
            f"{prediction.reason} {validation_reason}"
        )

    def build_plan(
        self,
        prediction: PredictionResult,
        option_chain_snapshot: OptionChainSnapshot | None,
        *,
        lot_size: int,
        max_lots: int,
        liquidity_score: float,
    ) -> StrategyPlan:
        """Build and validate a StrategyPlan."""

        spot_price = option_chain_snapshot.spot_price if option_chain_snapshot else prediction.expected_range_low or 0.0
        legs = self.generate_legs(
            prediction,
            option_chain_snapshot,
            lot_size=lot_size,
            max_lots=max_lots,
        )
        max_loss = self.estimate_max_loss(legs, spot_price or 1.0)
        valid, validation_reason = self.validate(
            prediction,
            liquidity_score=liquidity_score,
            max_loss=max_loss,
        )
        payoff = self.estimate_payoff(legs, spot_price or 1.0, prediction.expected_move_percent)
        return StrategyPlan(
            strategy_id=f"{self.strategy_type.value}-{now_ist().strftime('%Y%m%d%H%M%S')}",
            strategy_type=self.strategy_type,
            symbol=prediction.underlying or prediction.symbol,
            created_at=now_ist(),
            legs=legs if valid else (),
            confidence=prediction.confidence_score,
            rationale=self.explain_trade(prediction, validation_reason),
            max_loss=max_loss,
            expected_profit=payoff["estimated_max_profit"],
            metadata={
                "valid": valid,
                "validation_reason": validation_reason,
                "payoff": payoff,
                "breakeven": self.estimate_breakeven(legs, spot_price or 1.0),
                "advisory_only": True,
            },
        )

    @staticmethod
    def _leg(contract: OptionContract | None, side: OrderSide, quantity: int) -> StrategyLeg:
        if contract is None:
            raise ValueError("contract is required")
        price = contract.ask if side == OrderSide.BUY else contract.bid
        if price is None:
            price = contract.last_price
        return StrategyLeg(
            symbol=contract.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=price,
            metadata={
                "underlying": contract.underlying,
                "expiry": contract.expiry.isoformat(),
                "strike": contract.strike,
                "option_type": contract.option_type.upper(),
            },
        )

    @staticmethod
    def _nearest_atm(snapshot: OptionChainSnapshot, *, option_type: str) -> OptionContract | None:
        contracts = [contract for contract in snapshot.contracts if contract.option_type.upper() == option_type]
        if not contracts:
            return None
        return min(contracts, key=lambda contract: abs(contract.strike - snapshot.spot_price))

    @staticmethod
    def _nearest_otm(snapshot: OptionChainSnapshot, *, option_type: str) -> OptionContract | None:
        if option_type.upper() == "CE":
            contracts = [
                contract for contract in snapshot.contracts
                if contract.option_type.upper() == "CE" and contract.strike >= snapshot.spot_price
            ]
        else:
            contracts = [
                contract for contract in snapshot.contracts
                if contract.option_type.upper() == "PE" and contract.strike <= snapshot.spot_price
            ]
        if not contracts:
            return BaseStrategy._nearest_atm(snapshot, option_type=option_type)
        return min(contracts, key=lambda contract: abs(contract.strike - snapshot.spot_price))
