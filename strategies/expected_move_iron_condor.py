"""Expected-move iron condor research strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from models.market_models import OptionChainSnapshot, OptionContract
from strategies.expected_move_credit_spread import (
    _bid_ask_spread_percent,
    _clamp,
    _distribution_payoffs,
    _expected_move_band,
    _leg_payload,
    _max_loss_limit,
    _price_range,
    _read_value,
    _regime_name,
)
from strategies.option_strategy_utils import contracts_by_type, strike_width
from strategies.options_payoff import (
    calculate_breakevens,
    calculate_max_loss,
    calculate_max_profit,
    calculate_payoff_curve,
    iron_condor_payoff,
)
from strategies.strategy_research_base import BaseOptionsResearchStrategy, StrategyResearchResult


@dataclass
class ExpectedMoveIronCondorResearchStrategy(BaseOptionsResearchStrategy):
    """Research-only iron condor that sells outside an expected expiry range."""

    strategy_name: str = "ExpectedMoveIronCondor"
    min_probability_between: float = 0.65
    max_spread_width: float = 500.0
    max_bid_ask_spread_percent: float = 5.0
    min_liquidity_score: float = 0.35
    min_iv_percentile: float = 35.0
    allow_strong_trend: bool = False
    payoff_curve_points: int = 101

    def generate_candidates(
        self,
        *,
        option_chain_snapshot: OptionChainSnapshot,
        expiry_distribution: Any,
        regime: Any,
        news_risk: Any | None = None,
        iv_percentile: float | None = None,
        liquidity_score: float = 0.6,
        risk_limits: Any | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Generate defined-risk iron condor candidates."""

        del news_risk, iv_percentile, risk_limits
        lower_band, upper_band = _expected_move_band(expiry_distribution, option_chain_snapshot.spot_price)
        puts = contracts_by_type(option_chain_snapshot, "PE")
        calls = contracts_by_type(option_chain_snapshot, "CE")
        short_puts = [contract for contract in reversed(puts) if contract.strike <= lower_band]
        short_calls = [contract for contract in calls if contract.strike >= upper_band]
        candidates: list[dict[str, Any]] = []
        for short_put in short_puts:
            long_put = self._hedge_contract(option_chain_snapshot, short_put)
            if long_put is None:
                continue
            for short_call in short_calls:
                long_call = self._hedge_contract(option_chain_snapshot, short_call)
                if long_call is None:
                    continue
                put_width = strike_width(short_put, long_put)
                call_width = strike_width(short_call, long_call)
                probability_between = _probability_between(expiry_distribution, short_put.strike, short_call.strike)
                credit = _credit(short_put, long_put) + _credit(short_call, long_call)
                candidates.append(
                    {
                        "short_put": short_put,
                        "long_put": long_put,
                        "short_call": short_call,
                        "long_call": long_call,
                        "expiry_distribution": expiry_distribution,
                        "probability_between": probability_between,
                        "credit": credit,
                        "max_width": max(put_width, call_width),
                        "bid_ask_spread_percent": max(
                            _bid_ask_spread_percent(short_put),
                            _bid_ask_spread_percent(long_put),
                            _bid_ask_spread_percent(short_call),
                            _bid_ask_spread_percent(long_call),
                        ),
                        "liquidity_score": liquidity_score,
                        "regime": _regime_name(regime),
                    }
                )
        return candidates

    def score_candidate(self, candidate: Mapping[str, Any], **context: Any) -> StrategyResearchResult:
        """Score an iron condor candidate."""

        short_put = candidate["short_put"]
        long_put = candidate["long_put"]
        short_call = candidate["short_call"]
        long_call = candidate["long_call"]
        distribution = candidate["expiry_distribution"]
        quantity = int(context.get("quantity", short_put.lot_size or 1))
        credit = float(candidate["credit"])
        width = float(candidate["max_width"])
        max_profit = credit * quantity
        max_loss = max(width - credit, 0.0) * quantity
        price_range = _price_range(distribution, context.get("spot", short_put.strike), self.payoff_curve_points)
        kwargs = _payoff_kwargs(short_put, long_put, short_call, long_call, quantity)
        curve = calculate_payoff_curve(iron_condor_payoff, price_range, **kwargs)
        sample_payoffs = _distribution_payoffs(distribution, iron_condor_payoff, **kwargs)
        probability_of_profit = float(candidate["probability_between"])
        if sample_payoffs:
            expected_value = float(np.mean(sample_payoffs))
            probability_of_profit = float(np.mean(np.asarray(sample_payoffs) > 0))
            tail_loss = abs(float(np.quantile(sample_payoffs, 0.05)))
        else:
            expected_value = probability_of_profit * max_profit - (1.0 - probability_of_profit) * max_loss
            tail_loss = (1.0 - probability_of_profit) * max_loss
        risk_reward = max_profit / max_loss if max_loss > 0 else 0.0
        confidence = _clamp(
            0.20
            + probability_of_profit * 0.35
            + float(candidate.get("liquidity_score", 0.0)) * 0.15
            + min(max(float(context.get("iv_percentile", 50.0)), 0.0), 100.0) / 500.0,
            0.0,
            1.0,
        )
        result = StrategyResearchResult(
            strategy_name=self.strategy_name,
            legs=(
                _leg_payload(short_put, "SELL", quantity, _sell_price(short_put)),
                _leg_payload(long_put, "BUY", quantity, _buy_price(long_put)),
                _leg_payload(short_call, "SELL", quantity, _sell_price(short_call)),
                _leg_payload(long_call, "BUY", quantity, _buy_price(long_call)),
            ),
            expected_value=expected_value,
            probability_of_profit=probability_of_profit,
            max_profit=calculate_max_profit(curve) if curve else max_profit,
            max_loss=calculate_max_loss(curve) if curve else max_loss,
            breakevens=tuple(calculate_breakevens(curve)),
            payoff_curve=tuple(curve),
            tail_loss=tail_loss,
            risk_reward=risk_reward,
            confidence=confidence,
            reject_reason=None,
            explanation=(
                f"Sell put {short_put.strike:.0f} and call {short_call.strike:.0f} outside expected range; "
                f"estimated POP {probability_of_profit:.1%}, credit {credit:.2f}."
            ),
            metadata=dict(candidate),
        )
        reject_reason = self.reject_candidate(result, **context)
        if reject_reason is None:
            return result
        return _replace_reject_reason(result, reject_reason)

    def reject_candidate(self, result: StrategyResearchResult, **context: Any) -> str | None:
        """Reject iron condors that fail risk, regime, news, liquidity, or probability filters."""

        news_level = str(_read_value(context.get("news_risk"), "event_risk_level") or "").upper()
        if news_level in {"HIGH", "EXTREME"}:
            return f"News risk {news_level} blocks iron condor."
        regime_name = str(result.metadata.get("regime", "")).lower()
        if not self.allow_strong_trend and ("bullish" in regime_name or "bearish" in regime_name):
            return "Strong trending regime blocks iron condor."
        if result.expected_value < 0:
            return "Expected value is negative."
        if result.probability_of_profit < self.min_probability_between:
            return "Probability between short strikes is below threshold."
        if result.max_loss <= 0:
            return "Max loss is undefined."
        if float(result.metadata.get("max_width", 0.0)) > self.max_spread_width:
            return "Spread width exceeds configured maximum."
        if float(result.metadata.get("bid_ask_spread_percent", 0.0)) > self.max_bid_ask_spread_percent:
            return "Bid-ask spread is too wide."
        if float(result.metadata.get("liquidity_score", 0.0)) < self.min_liquidity_score:
            return "Liquidity score is too low."
        iv_percentile = float(context.get("iv_percentile", 50.0))
        if iv_percentile < self.min_iv_percentile:
            return "IV percentile is too low for iron condor premium selling."
        risk_limit = _max_loss_limit(context.get("risk_limits"))
        if risk_limit is not None and result.max_loss > risk_limit:
            return "Max loss exceeds risk limit."
        return None

    def analyze(
        self,
        *,
        option_chain_snapshot: OptionChainSnapshot,
        expiry_distribution: Any,
        regime: Any,
        news_risk: Any | None = None,
        iv_percentile: float | None = None,
        liquidity_score: float = 0.6,
        risk_limits: Any | None = None,
        quantity: int | None = None,
    ) -> StrategyResearchResult | None:
        """Generate, score, and select the best iron condor candidate."""

        candidates = self.generate_candidates(
            option_chain_snapshot=option_chain_snapshot,
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
            spot=option_chain_snapshot.spot_price,
        )

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


def _probability_between(distribution: Any, lower: float, upper: float) -> float:
    method = getattr(distribution, "probability_between", None)
    if callable(method):
        return float(method(lower, upper))
    samples = np.asarray(_read_value(distribution, "samples") or (), dtype=float)
    if samples.size:
        return float(np.mean((samples >= lower) & (samples <= upper)))
    return 0.0


def _payoff_kwargs(
    short_put: OptionContract,
    long_put: OptionContract,
    short_call: OptionContract,
    long_call: OptionContract,
    quantity: int,
) -> dict[str, float | int]:
    return {
        "short_put_strike": short_put.strike,
        "long_put_strike": long_put.strike,
        "short_call_strike": short_call.strike,
        "long_call_strike": long_call.strike,
        "short_put_premium": _sell_price(short_put),
        "long_put_premium": _buy_price(long_put),
        "short_call_premium": _sell_price(short_call),
        "long_call_premium": _buy_price(long_call),
        "quantity": quantity,
    }


def _credit(short_contract: OptionContract, long_contract: OptionContract) -> float:
    return _sell_price(short_contract) - _buy_price(long_contract)


def _sell_price(contract: OptionContract) -> float:
    return float(contract.bid or contract.last_price or 0.0)


def _buy_price(contract: OptionContract) -> float:
    return float(contract.ask or contract.last_price or 0.0)


def _replace_reject_reason(result: StrategyResearchResult, reason: str) -> StrategyResearchResult:
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
        reject_reason=reason,
        explanation=result.explanation,
        metadata=result.metadata,
    )
