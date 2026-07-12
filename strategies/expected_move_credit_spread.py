"""Expected-move based defined-risk credit spread strategy."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from models import OrderSide, PredictionResult, StrategyPlan, StrategyType
from models.market_models import OptionChainSnapshot, OptionContract
from strategies.base_strategy import BaseStrategy
from strategies.option_strategy_utils import (
    StrategyMetrics,
    contracts_by_type,
    make_leg,
    missing_chain_plan_reason,
    net_premium,
    strike_width,
)
from strategies.options_payoff import (
    bear_call_credit_spread_payoff,
    bull_put_credit_spread_payoff,
    calculate_breakevens,
    calculate_max_loss,
    calculate_max_profit,
    calculate_payoff_curve,
)
from strategies.strategy_research_base import BaseOptionsResearchStrategy, StrategyResearchResult
from utils.time_utils import now_ist

SpreadVariant = Literal["bull_put", "bear_call"]


@dataclass
class ExpectedMoveCreditSpreadResearchStrategy(BaseOptionsResearchStrategy):
    """Research-only expected-move credit spread selector."""

    strategy_name: str = "ExpectedMoveCreditSpread"
    min_probability_otm: float = 0.65
    max_spread_width: float = 500.0
    max_bid_ask_spread_percent: float = 5.0
    min_liquidity_score: float = 0.35
    min_risk_reward: float = 0.10
    payoff_curve_points: int = 81

    def generate_candidates(
        self,
        *,
        option_chain_snapshot: OptionChainSnapshot,
        spot: float | None = None,
        expiry_distribution: Any,
        regime: Any,
        news_risk: Any | None = None,
        iv_percentile: float | None = None,
        liquidity_score: float = 0.6,
        risk_limits: Any | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Generate candidate bull-put and bear-call credit spreads."""

        del news_risk, iv_percentile, risk_limits
        spot_price = float(spot or option_chain_snapshot.spot_price)
        lower_band, upper_band = _expected_move_band(expiry_distribution, spot_price)
        variants = self._variants_for_regime(regime)
        candidates: list[dict[str, Any]] = []
        for variant in variants:
            contracts = contracts_by_type(option_chain_snapshot, "PE" if variant == "bull_put" else "CE")
            shorts = self._short_candidates(contracts, variant, lower_band, upper_band)
            for short_contract in shorts:
                probability_otm = _probability_otm(expiry_distribution, variant, short_contract.strike)
                long_contract = self._hedge_contract(option_chain_snapshot, short_contract)
                if long_contract is None:
                    continue
                width = strike_width(short_contract, long_contract)
                short_spread = _bid_ask_spread_percent(short_contract)
                long_spread = _bid_ask_spread_percent(long_contract)
                credit = _contract_credit(short_contract, long_contract)
                candidates.append(
                    {
                        "variant": variant,
                        "short_contract": short_contract,
                        "long_contract": long_contract,
                        "spot": spot_price,
                        "expiry_distribution": expiry_distribution,
                        "probability_otm": probability_otm,
                        "credit": credit,
                        "spread_width": width,
                        "bid_ask_spread_percent": max(short_spread, long_spread),
                        "liquidity_score": liquidity_score,
                        "regime": _regime_name(regime),
                        "regime_compatible": self._regime_compatible(variant, regime),
                    }
                )
        return candidates

    def score_candidate(self, candidate: Mapping[str, Any], **context: Any) -> StrategyResearchResult:
        """Score a credit-spread candidate using distribution-aware payoff statistics."""

        variant = str(candidate["variant"])
        short_contract = candidate["short_contract"]
        long_contract = candidate["long_contract"]
        spot = float(candidate["spot"])
        distribution = candidate["expiry_distribution"]
        quantity = int(context.get("quantity", short_contract.lot_size or 1))
        credit = float(candidate["credit"])
        width = float(candidate["spread_width"])
        max_profit = credit * quantity
        max_loss = max(width - credit, 0.0) * quantity
        breakeven = short_contract.strike - credit if variant == "bull_put" else short_contract.strike + credit
        price_range = _price_range(distribution, spot, self.payoff_curve_points)
        payoff_function = bull_put_credit_spread_payoff if variant == "bull_put" else bear_call_credit_spread_payoff
        curve = calculate_payoff_curve(
            payoff_function,
            price_range,
            short_strike=short_contract.strike,
            long_strike=long_contract.strike,
            short_premium=float(short_contract.bid or short_contract.last_price or 0.0),
            long_premium=float(long_contract.ask or long_contract.last_price or 0.0),
            quantity=quantity,
        )
        sample_payoffs = _distribution_payoffs(
            distribution,
            payoff_function,
            short_strike=short_contract.strike,
            long_strike=long_contract.strike,
            short_premium=float(short_contract.bid or short_contract.last_price or 0.0),
            long_premium=float(long_contract.ask or long_contract.last_price or 0.0),
            quantity=quantity,
        )
        if sample_payoffs:
            expected_value = float(sum(sample_payoffs) / len(sample_payoffs))
            probability_of_profit = float(sum(1 for payoff in sample_payoffs if payoff > 0) / len(sample_payoffs))
            tail_loss = abs(float(np.quantile(sample_payoffs, 0.05)))
        else:
            probability_of_profit = _probability_otm(distribution, variant, breakeven)
            expected_value = probability_of_profit * max_profit - (1.0 - probability_of_profit) * max_loss
            tail_loss = max_loss * (1.0 - probability_of_profit)
        risk_reward = max_profit / max_loss if max_loss > 0 else 0.0
        confidence = _clamp(
            0.25
            + probability_of_profit * 0.35
            + float(candidate.get("liquidity_score", 0.0)) * 0.15
            + (0.15 if candidate.get("regime_compatible") else 0.0)
            + min(max(float(context.get("iv_percentile", 50.0)), 0.0), 100.0) / 1000.0,
            0.0,
            1.0,
        )
        legs = (
            _leg_payload(short_contract, "SELL", quantity, float(short_contract.bid or short_contract.last_price or 0.0)),
            _leg_payload(long_contract, "BUY", quantity, float(long_contract.ask or long_contract.last_price or 0.0)),
        )
        result = StrategyResearchResult(
            strategy_name=f"{self.strategy_name}:{variant}",
            legs=legs,
            expected_value=expected_value,
            probability_of_profit=probability_of_profit,
            max_profit=calculate_max_profit(curve) if curve else max_profit,
            max_loss=calculate_max_loss(curve) if curve else max_loss,
            breakevens=tuple(calculate_breakevens(curve)) or (breakeven,),
            payoff_curve=tuple(curve),
            tail_loss=tail_loss,
            risk_reward=risk_reward,
            confidence=confidence,
            reject_reason=None,
            explanation=(
                f"{variant} short strike {short_contract.strike:.0f} has "
                f"{probability_of_profit:.1%} estimated profit probability with credit {credit:.2f}."
            ),
            metadata=dict(candidate),
        )
        reject_reason = self.reject_candidate(result, **context)
        if reject_reason is None:
            return result
        return StrategyResearchResult(
            strategy_name=result.strategy_name,
            legs=result.legs,
            expected_value=result.expected_value,
            probability_of_profit=result.probability_of_profit,
            max_profit=result.max_profit,
            max_loss=result.max_loss,
            breakevens=result.breakevens,
            payoff_curve=result.payoff_curve,
            tail_loss=result.tail_loss,
            risk_reward=result.risk_reward,
            confidence=result.confidence,
            reject_reason=reject_reason,
            explanation=result.explanation,
            metadata=result.metadata,
        )

    def reject_candidate(self, result: StrategyResearchResult, **context: Any) -> str | None:
        """Reject candidates that fail EV, POP, risk, news, liquidity, or spread filters."""

        news_level = str(_read_value(context.get("news_risk"), "event_risk_level") or "").upper()
        if news_level in {"HIGH", "EXTREME"}:
            return f"News risk {news_level} blocks credit-spread research candidate."
        if result.expected_value < 0:
            return "Expected value is negative."
        if result.probability_of_profit < self.min_probability_otm:
            return "Probability of profit is below threshold."
        if result.risk_reward < self.min_risk_reward:
            return "Risk reward is below threshold."
        if float(result.metadata.get("spread_width", 0.0)) > self.max_spread_width:
            return "Spread width exceeds configured maximum."
        if float(result.metadata.get("bid_ask_spread_percent", 0.0)) > self.max_bid_ask_spread_percent:
            return "Bid-ask spread is too wide."
        if float(result.metadata.get("liquidity_score", 0.0)) < self.min_liquidity_score:
            return "Liquidity score is too low."
        risk_limit = _max_loss_limit(context.get("risk_limits"))
        if risk_limit is not None and result.max_loss > risk_limit:
            return "Max loss exceeds risk limit."
        return None

    def analyze(
        self,
        *,
        option_chain_snapshot: OptionChainSnapshot,
        spot: float | None = None,
        expiry_distribution: Any,
        regime: Any,
        news_risk: Any | None = None,
        iv_percentile: float | None = None,
        liquidity_score: float = 0.6,
        risk_limits: Any | None = None,
        quantity: int | None = None,
    ) -> StrategyResearchResult | None:
        """Generate, score, and select the best research candidate."""

        candidates = self.generate_candidates(
            option_chain_snapshot=option_chain_snapshot,
            spot=spot,
            expiry_distribution=expiry_distribution,
            regime=regime,
            news_risk=news_risk,
            iv_percentile=iv_percentile,
            liquidity_score=liquidity_score,
            risk_limits=risk_limits,
        )
        return self.select_best_candidate(
            candidates,
            news_risk=news_risk,
            iv_percentile=iv_percentile if iv_percentile is not None else 50.0,
            risk_limits=risk_limits,
            quantity=quantity or option_chain_snapshot.contracts[0].lot_size if option_chain_snapshot.contracts else 1,
        )

    def _short_candidates(
        self,
        contracts: Sequence[OptionContract],
        variant: str,
        lower_band: float,
        upper_band: float,
    ) -> list[OptionContract]:
        if variant == "bull_put":
            return [contract for contract in reversed(contracts) if contract.strike <= lower_band]
        return [contract for contract in contracts if contract.strike >= upper_band]

    def _hedge_contract(self, snapshot: OptionChainSnapshot, short_contract: OptionContract) -> OptionContract | None:
        contracts = contracts_by_type(snapshot, short_contract.option_type)
        if short_contract.option_type.upper() == "PE":
            candidates = [contract for contract in reversed(contracts) if contract.strike < short_contract.strike]
        else:
            candidates = [contract for contract in contracts if contract.strike > short_contract.strike]
        for candidate in candidates:
            if strike_width(short_contract, candidate) <= self.max_spread_width:
                return candidate
        return None

    @staticmethod
    def _variants_for_regime(regime: Any) -> tuple[SpreadVariant, ...]:
        name = _regime_name(regime).lower()
        if "bull" in name:
            return ("bull_put",)
        if "bear" in name:
            return ("bear_call",)
        if "sideways" in name or "range" in name:
            return ("bull_put", "bear_call")
        return ("bull_put", "bear_call")

    @staticmethod
    def _regime_compatible(variant: str, regime: Any) -> bool:
        name = _regime_name(regime).lower()
        if "bull" in name:
            return variant == "bull_put"
        if "bear" in name:
            return variant == "bear_call"
        return "sideways" in name or "range" in name or not name


@dataclass
class ExpectedMoveCreditSpreadStrategy(BaseStrategy):
    """Select a credit spread whose short strike is outside expected expiry range."""

    name: str = "Expected Move Credit Spread"
    strategy_type: StrategyType = StrategyType.EXPECTED_MOVE_CREDIT_SPREAD
    allowed_regimes: tuple[str, ...] = ("trending", "range_bound", "high_volatility", "low_volatility")
    forbidden_regimes: tuple[str, ...] = ()
    required_confidence: float = 0.65
    min_probability_otm: float = 0.68
    min_reward_risk: float = 0.20
    max_spread_width: float = 500.0
    max_bid_ask_spread_percent: float = 5.0
    confidence_interval_z: float = 1.0
    expected_shortfall_tail_percent: float = 5.0
    min_iv_percentile: float = 35.0

    def build_plan(
        self,
        prediction: PredictionResult,
        option_chain_snapshot: OptionChainSnapshot | None,
        *,
        lot_size: int,
        max_lots: int,
        liquidity_score: float,
    ) -> StrategyPlan:
        """Build a defined-risk spread plan from expected move and option-chain quotes."""

        reason = missing_chain_plan_reason(option_chain_snapshot)
        if reason:
            return _blocked_plan(self, prediction, reason)
        assert option_chain_snapshot is not None

        news_level = str(prediction.metadata.get("event_risk_level") or "").upper()
        if news_level in {"HIGH", "EXTREME"}:
            return _blocked_plan(self, prediction, f"News event risk {news_level} blocks credit spread selling.")

        variant = self._variant(prediction)
        distribution = self._expiry_distribution(prediction, option_chain_snapshot)
        lower_bound, upper_bound = self._confidence_interval(distribution, prediction, option_chain_snapshot)
        selected = self._select_contracts(option_chain_snapshot, distribution, variant)
        if selected is None:
            return _blocked_plan(self, prediction, f"No {variant} spread met probability and liquidity filters.")

        short_contract, long_contract, probability_otm = selected
        quantity = lot_size * max(1, max_lots)
        legs = (
            make_leg(short_contract, OrderSide.SELL, quantity),
            make_leg(long_contract, OrderSide.BUY, quantity),
        )
        credit = net_premium(legs)
        width = strike_width(short_contract, long_contract)
        if credit <= 0:
            return _blocked_plan(self, prediction, "Selected spread has no net credit.")

        max_profit = credit * quantity
        max_loss = max(width - credit, 0.0) * quantity
        if max_loss <= 0:
            return _blocked_plan(self, prediction, "Selected spread has invalid maximum loss.")

        breakeven = short_contract.strike - credit if variant == "bull_put" else short_contract.strike + credit
        probability_of_profit = self._probability_of_profit(distribution, variant, breakeven)
        expected_shortfall = (1.0 - probability_of_profit) * max_loss
        expected_value = probability_of_profit * max_profit - (1.0 - probability_of_profit) * max_loss
        reward_risk = max_profit / max_loss if max_loss else 0.0

        rejection = self._rejection_reason(
            expected_value=expected_value,
            reward_risk=reward_risk,
            width=width,
            short_contract=short_contract,
            long_contract=long_contract,
        )
        if rejection:
            return _blocked_plan(self, prediction, rejection)

        valid, validation_reason = self.validate(prediction, liquidity_score=liquidity_score, max_loss=max_loss)
        metrics = StrategyMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=(breakeven,),
            target_profit=max_profit * 0.5,
            stop_loss=max_loss * 0.5,
            margin_estimate_placeholder=max_loss,
        )
        metadata = {
            "valid": valid,
            "validation_reason": validation_reason,
            "variant": variant,
            "expected_expiry_price": distribution.mean,
            "confidence_interval_low": lower_bound,
            "confidence_interval_high": upper_bound,
            "short_strike_probability_otm": probability_otm,
            "probability_of_profit": probability_of_profit,
            "expected_value": expected_value,
            "expected_shortfall": expected_shortfall,
            "reward_risk": reward_risk,
            "spread_width": width,
            "net_credit": credit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": (breakeven,),
            "target_profit": metrics.target_profit,
            "stop_loss": metrics.stop_loss,
            "entry_reason": self._entry_reason(variant, probability_otm),
            "exit_conditions": (
                "Exit at 50% of credit captured.",
                "Exit if loss reaches configured stop.",
                "Exit if spot crosses breakeven or short strike risk increases.",
                "Exit before expiry if news or liquidity risk rises.",
            ),
            "margin_estimate_placeholder": metrics.margin_estimate_placeholder,
            "metrics": metrics.as_dict(),
            "advisory_only": True,
        }
        return StrategyPlan(
            strategy_id=f"expected-move-credit-spread-{now_ist().strftime('%Y%m%d%H%M%S')}",
            strategy_type=self.strategy_type,
            symbol=prediction.underlying or prediction.symbol,
            created_at=now_ist(),
            legs=legs if valid else (),
            confidence=prediction.confidence_score,
            rationale=self.explain_trade(prediction, validation_reason),
            max_loss=max_loss,
            expected_profit=max_profit,
            metadata=metadata,
        )

    def _select_contracts(
        self,
        snapshot: OptionChainSnapshot,
        distribution: NormalDist,
        variant: SpreadVariant,
    ) -> tuple[OptionContract, OptionContract, float] | None:
        option_type = "PE" if variant == "bull_put" else "CE"
        contracts = contracts_by_type(snapshot, option_type)
        if variant == "bull_put":
            candidates = [contract for contract in reversed(contracts) if contract.strike < snapshot.spot_price]
        else:
            candidates = [contract for contract in contracts if contract.strike > snapshot.spot_price]

        best: tuple[OptionContract, OptionContract, float, float] | None = None
        for short_contract in candidates:
            probability_otm = self._probability_otm(distribution, variant, short_contract.strike)
            if probability_otm < self.min_probability_otm:
                continue
            if self._bid_ask_spread_percent(short_contract) > self.max_bid_ask_spread_percent + 1e-9:
                continue
            long_contract = self._hedge_contract(snapshot, short_contract)
            if long_contract is None:
                continue
            width = strike_width(short_contract, long_contract)
            if width > self.max_spread_width:
                continue
            if self._bid_ask_spread_percent(long_contract) > self.max_bid_ask_spread_percent + 1e-9:
                continue
            credit = self._contract_credit(short_contract, long_contract)
            if best is None or credit > best[3]:
                best = (short_contract, long_contract, probability_otm, credit)
        if best is None:
            return None
        return best[0], best[1], best[2]

    def _rejection_reason(
        self,
        *,
        expected_value: float,
        reward_risk: float,
        width: float,
        short_contract: OptionContract,
        long_contract: OptionContract,
    ) -> str | None:
        if expected_value < 0:
            return "Expected value is negative."
        if reward_risk < self.min_reward_risk:
            return "Risk reward is below configured threshold."
        if width > self.max_spread_width:
            return "Spread width exceeds configured maximum."
        max_spread = max(self._bid_ask_spread_percent(short_contract), self._bid_ask_spread_percent(long_contract))
        if max_spread > self.max_bid_ask_spread_percent + 1e-9:
            return "Bid-ask spread is too wide."
        return None

    def _expiry_distribution(self, prediction: PredictionResult, snapshot: OptionChainSnapshot) -> NormalDist:
        payload = prediction.metadata.get("expected_expiry_distribution")
        if isinstance(payload, Mapping):
            mean = _optional_float(payload.get("mean"))
            stdev = _optional_float(payload.get("stdev") or payload.get("std") or payload.get("sigma"))
            if mean is not None and stdev is not None and stdev > 0:
                return NormalDist(mu=mean, sigma=stdev)

        expected_return = prediction.expected_return_percent or 0.0
        mean = snapshot.spot_price * (1.0 + expected_return / 100.0)
        stdev = max(snapshot.spot_price * prediction.expected_move_percent / 100.0, snapshot.spot_price * 0.001)
        return NormalDist(mu=mean, sigma=stdev)

    def _confidence_interval(
        self,
        distribution: NormalDist,
        prediction: PredictionResult,
        snapshot: OptionChainSnapshot,
    ) -> tuple[float, float]:
        expected_move_points = snapshot.spot_price * max(prediction.expected_move_percent, 0.0) / 100.0
        width = max(self.confidence_interval_z * distribution.stdev, expected_move_points)
        return distribution.mean - width, distribution.mean + width

    def _probability_otm(self, distribution: NormalDist, variant: SpreadVariant, strike: float) -> float:
        if variant == "bull_put":
            return 1.0 - distribution.cdf(strike)
        return distribution.cdf(strike)

    def _probability_of_profit(self, distribution: NormalDist, variant: SpreadVariant, breakeven: float) -> float:
        if variant == "bull_put":
            return 1.0 - distribution.cdf(breakeven)
        return distribution.cdf(breakeven)

    def _hedge_contract(self, snapshot: OptionChainSnapshot, short_contract: OptionContract) -> OptionContract | None:
        contracts = contracts_by_type(snapshot, short_contract.option_type)
        if short_contract.option_type.upper() == "PE":
            candidates = [contract for contract in reversed(contracts) if contract.strike < short_contract.strike]
        else:
            candidates = [contract for contract in contracts if contract.strike > short_contract.strike]
        return candidates[0] if candidates else None

    def _variant(self, prediction: PredictionResult) -> SpreadVariant:
        if prediction.bullish_probability >= prediction.bearish_probability:
            return "bull_put"
        return "bear_call"

    def _entry_reason(self, variant: SpreadVariant, probability_otm: float) -> str:
        label = "bull put" if variant == "bull_put" else "bear call"
        return (
            f"Expected-move {label} spread selected because short strike has "
            f"{probability_otm:.1%} estimated probability of expiring OTM."
        )

    @staticmethod
    def _contract_credit(short_contract: OptionContract, long_contract: OptionContract) -> float:
        short_price = float(short_contract.bid or short_contract.last_price or 0.0)
        long_price = float(long_contract.ask or long_contract.last_price or 0.0)
        return short_price - long_price

    @staticmethod
    def _bid_ask_spread_percent(contract: OptionContract) -> float:
        bid = _optional_float(contract.bid)
        ask = _optional_float(contract.ask)
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return 0.0
        mid = (bid + ask) / 2.0
        return 100.0 * (ask - bid) / mid if mid > 0 else 100.0


def _blocked_plan(strategy: ExpectedMoveCreditSpreadStrategy, prediction: PredictionResult, reason: str) -> StrategyPlan:
    return StrategyPlan(
        strategy_id=f"expected-move-credit-spread-blocked-{now_ist().strftime('%Y%m%d%H%M%S')}",
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


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expected_move_band(expiry_distribution: Any, spot: float) -> tuple[float, float]:
    lower = _read_value(expiry_distribution, "lower_bound_68")
    upper = _read_value(expiry_distribution, "upper_bound_68")
    if lower is not None and upper is not None:
        return float(lower), float(upper)
    move = _optional_float(_read_value(expiry_distribution, "expected_move_points")) or spot * 0.01
    return spot - move, spot + move


def _probability_otm(expiry_distribution: Any, variant: str, strike: float) -> float:
    if variant == "bull_put":
        method = getattr(expiry_distribution, "probability_above_strike", None)
    else:
        method = getattr(expiry_distribution, "probability_below_strike", None)
    if callable(method):
        return float(method(strike))
    samples = _samples(expiry_distribution)
    if samples.size:
        if variant == "bull_put":
            return float(np.mean(samples > strike))
        return float(np.mean(samples < strike))
    return 0.0


def _price_range(expiry_distribution: Any, spot: float, points: int) -> np.ndarray:
    lower = _read_value(expiry_distribution, "lower_bound_95")
    upper = _read_value(expiry_distribution, "upper_bound_95")
    if lower is None or upper is None:
        move = _optional_float(_read_value(expiry_distribution, "expected_move_points")) or spot * 0.02
        lower = spot - move * 2.0
        upper = spot + move * 2.0
    return np.linspace(float(lower), float(upper), max(11, int(points)))


def _distribution_payoffs(expiry_distribution: Any, payoff_function: Any, **kwargs: Any) -> list[float]:
    samples = _samples(expiry_distribution)
    if not samples.size:
        return []
    return [float(payoff_function(float(price), **kwargs)) for price in samples]


def _samples(expiry_distribution: Any) -> np.ndarray:
    raw_samples = _read_value(expiry_distribution, "samples")
    if raw_samples is None:
        return np.array([], dtype=float)
    samples = np.asarray(raw_samples, dtype=float)
    return samples[np.isfinite(samples)]


def _leg_payload(contract: OptionContract, side: str, quantity: int, premium: float) -> dict[str, Any]:
    return {
        "symbol": contract.symbol,
        "side": side,
        "option_type": contract.option_type.upper(),
        "strike": contract.strike,
        "premium": premium,
        "quantity": quantity,
        "bid": contract.bid,
        "ask": contract.ask,
        "last_price": contract.last_price,
    }


def _max_loss_limit(risk_limits: Any) -> float | None:
    if risk_limits is None:
        return None
    explicit = _optional_float(_read_value(risk_limits, "max_loss_amount"))
    if explicit is not None:
        return explicit
    explicit = _optional_float(_read_value(risk_limits, "max_loss_per_trade_amount"))
    if explicit is not None:
        return explicit
    capital = _optional_float(_read_value(risk_limits, "capital"))
    percent = _optional_float(_read_value(risk_limits, "max_loss_per_trade_percent"))
    if capital is not None and percent is not None:
        return capital * percent / 100.0
    return None


def _regime_name(regime: Any) -> str:
    if regime is None:
        return ""
    if isinstance(regime, str):
        return regime
    return str(getattr(regime, "regime_name", None) or getattr(getattr(regime, "regime", ""), "value", "") or regime)


def _read_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _contract_credit(short_contract: OptionContract, long_contract: OptionContract) -> float:
    return float(short_contract.bid or short_contract.last_price or 0.0) - float(
        long_contract.ask or long_contract.last_price or 0.0
    )


def _bid_ask_spread_percent(contract: OptionContract) -> float:
    bid = _optional_float(contract.bid)
    ask = _optional_float(contract.ask)
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return 100.0
    mid = (bid + ask) / 2.0
    return 100.0 * (ask - bid) / mid if mid > 0 else 100.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
