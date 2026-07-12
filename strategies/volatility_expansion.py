"""Volatility expansion long-premium research strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from models.market_models import OptionChainSnapshot, OptionContract
from strategies.expected_move_credit_spread import _leg_payload, _max_loss_limit, _price_range, _read_value
from strategies.option_strategy_utils import atm_contract, contracts_by_type
from strategies.options_payoff import (
    bull_call_spread_payoff,
    calculate_breakevens,
    calculate_max_loss,
    calculate_max_profit,
    calculate_payoff_curve,
    long_straddle_payoff,
)
from strategies.strategy_research_base import BaseOptionsResearchStrategy, StrategyResearchResult


@dataclass
class VolatilityExpansionResearchStrategy(BaseOptionsResearchStrategy):
    """Research-only long premium strategy for expected volatility expansion."""

    strategy_name: str = "VolatilityExpansion"
    max_iv_percentile_for_straddle: float = 70.0
    max_iv_percentile: float = 85.0
    min_breakout_probability: float = 0.55
    min_forecast_rise: float = 0.03

    def generate_candidates(
        self,
        *,
        option_chain_snapshot: OptionChainSnapshot,
        expiry_distribution: Any,
        realized_volatility: float,
        forecast_volatility: float,
        breakout_probability: float,
        news_risk: Any | None = None,
        iv_percentile: float = 50.0,
        risk_limits: Any | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Generate long straddle and debit-spread candidates."""

        del news_risk, risk_limits
        call = atm_contract(option_chain_snapshot, "CE")
        put = atm_contract(option_chain_snapshot, "PE")
        candidates: list[dict[str, Any]] = []
        if call and put:
            candidates.append(
                {
                    "structure": "long_straddle",
                    "call": call,
                    "put": put,
                    "expiry_distribution": expiry_distribution,
                    "realized_volatility": realized_volatility,
                    "forecast_volatility": forecast_volatility,
                    "breakout_probability": breakout_probability,
                    "iv_percentile": iv_percentile,
                }
            )
        debit = _debit_spread_contracts(option_chain_snapshot)
        if debit:
            long_call, short_call = debit
            candidates.append(
                {
                    "structure": "debit_spread",
                    "long_call": long_call,
                    "short_call": short_call,
                    "expiry_distribution": expiry_distribution,
                    "realized_volatility": realized_volatility,
                    "forecast_volatility": forecast_volatility,
                    "breakout_probability": breakout_probability,
                    "iv_percentile": iv_percentile,
                }
            )
        return candidates

    def score_candidate(self, candidate: Mapping[str, Any], **context: Any) -> StrategyResearchResult:
        """Score long-premium candidate against expected move."""

        quantity = int(context.get("quantity", _candidate_lot_size(candidate)))
        distribution = candidate["expiry_distribution"]
        structure = str(candidate["structure"])
        spot = float(context["option_chain_snapshot"].spot_price)
        price_range = _price_range(distribution, spot, 101)
        if structure == "long_straddle":
            call = candidate["call"]
            put = candidate["put"]
            cost = _buy_price(call) + _buy_price(put)
            kwargs = {"strike": call.strike, "call_premium": _buy_price(call), "put_premium": _buy_price(put), "quantity": quantity}
            curve = calculate_payoff_curve(long_straddle_payoff, price_range, **kwargs)
            legs = (
                _leg_payload(call, "BUY", quantity, _buy_price(call)),
                _leg_payload(put, "BUY", quantity, _buy_price(put)),
            )
        else:
            long_call = candidate["long_call"]
            short_call = candidate["short_call"]
            cost = _buy_price(long_call) - _sell_price(short_call)
            kwargs = {
                "long_strike": long_call.strike,
                "short_strike": short_call.strike,
                "long_premium": _buy_price(long_call),
                "short_premium": _sell_price(short_call),
                "quantity": quantity,
            }
            curve = calculate_payoff_curve(bull_call_spread_payoff, price_range, **kwargs)
            legs = (
                _leg_payload(long_call, "BUY", quantity, _buy_price(long_call)),
                _leg_payload(short_call, "SELL", quantity, _sell_price(short_call)),
            )
        samples = np.asarray(_read_value(distribution, "samples") or (), dtype=float)
        if samples.size:
            payoffs = np.asarray([row["payoff"] for row in calculate_payoff_curve(
                long_straddle_payoff if structure == "long_straddle" else bull_call_spread_payoff,
                samples,
                **kwargs,
            )])
            expected_value = float(np.mean(payoffs))
            probability_of_profit = float(np.mean(payoffs > 0))
            tail_loss = abs(float(np.quantile(payoffs, 0.05)))
        else:
            expected_move = float(_read_value(distribution, "expected_move_points") or 0.0)
            expected_value = (expected_move - cost) * quantity
            probability_of_profit = float(candidate.get("breakout_probability", 0.0))
            tail_loss = cost * quantity
        result = StrategyResearchResult(
            strategy_name=f"{self.strategy_name}:{structure}",
            legs=legs,
            expected_value=expected_value,
            probability_of_profit=probability_of_profit,
            max_profit=calculate_max_profit(curve),
            max_loss=calculate_max_loss(curve),
            breakevens=tuple(calculate_breakevens(curve)),
            payoff_curve=tuple(curve),
            tail_loss=tail_loss,
            risk_reward=calculate_max_profit(curve) / calculate_max_loss(curve) if calculate_max_loss(curve) > 0 else 0.0,
            confidence=min(1.0, 0.25 + probability_of_profit * 0.35 + float(candidate["breakout_probability"]) * 0.25),
            reject_reason=None,
            explanation=f"{structure} cost {cost:.2f}; expected value {expected_value:.2f}.",
            metadata=dict(candidate),
        )
        return _with_reject_reason(result, self.reject_candidate(result, **context))

    def reject_candidate(self, result: StrategyResearchResult, **context: Any) -> str | None:
        """Reject unsuitable volatility expansion candidates."""

        metadata = result.metadata
        news_level = str(_read_value(context.get("news_risk"), "event_risk_level") or "LOW").upper()
        if news_level == "EXTREME":
            return "EXTREME news risk blocks volatility expansion."
        if float(metadata.get("breakout_probability", 0.0)) < self.min_breakout_probability:
            return "Breakout probability is too low."
        spread = _as_decimal(float(metadata.get("forecast_volatility", 0.0))) - _as_decimal(float(metadata.get("realized_volatility", 0.0)))
        if spread < self.min_forecast_rise:
            return "Forecast volatility is not rising enough."
        iv_percentile = float(metadata.get("iv_percentile", 50.0))
        if iv_percentile > self.max_iv_percentile:
            return "IV is too expensive for long premium."
        if result.strategy_name.endswith("long_straddle") and iv_percentile > self.max_iv_percentile_for_straddle:
            return "IV is too high for long straddle."
        risk_limit = _max_loss_limit(context.get("risk_limits"))
        if risk_limit is not None and result.max_loss > risk_limit:
            return "Max loss exceeds risk limit."
        if result.expected_value < 0:
            return "Expected value is negative."
        return None

    def analyze(self, **kwargs: Any) -> StrategyResearchResult | None:
        """Select best volatility expansion candidate."""

        return self.select_best_candidate(self.generate_candidates(**kwargs), **kwargs)


def _candidate_lot_size(candidate: Mapping[str, Any]) -> int:
    for key in ("call", "long_call"):
        contract = candidate.get(key)
        if contract is not None:
            return int(contract.lot_size)
    return 1


def _debit_spread_contracts(snapshot: OptionChainSnapshot) -> tuple[OptionContract, OptionContract] | None:
    calls = contracts_by_type(snapshot, "CE")
    long_candidates = [contract for contract in calls if contract.strike >= snapshot.spot_price]
    if len(long_candidates) < 2:
        return None
    return long_candidates[0], long_candidates[1]


def _buy_price(contract: OptionContract) -> float:
    return float(contract.ask or contract.last_price or 0.0)


def _sell_price(contract: OptionContract) -> float:
    return float(contract.bid or contract.last_price or 0.0)


def _as_decimal(value: float) -> float:
    value = abs(float(value))
    return value / 100.0 if value > 1.0 else value


def _with_reject_reason(result: StrategyResearchResult, reason: str | None) -> StrategyResearchResult:
    return StrategyResearchResult(
        result.strategy_name,
        result.legs,
        result.expected_value,
        result.probability_of_profit,
        result.max_profit,
        result.max_loss,
        result.breakevens,
        result.payoff_curve,
        result.tail_loss,
        result.risk_reward,
        result.confidence,
        reason,
        result.explanation,
        result.metadata,
    )
