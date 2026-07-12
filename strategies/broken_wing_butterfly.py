"""Broken wing butterfly research strategy."""

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
from strategies.expected_move_iron_condor import _buy_price, _replace_reject_reason, _sell_price
from strategies.option_strategy_utils import contracts_by_type
from strategies.options_payoff import (
    broken_wing_butterfly_payoff,
    calculate_breakevens,
    calculate_max_loss,
    calculate_max_profit,
    calculate_payoff_curve,
)
from strategies.strategy_research_base import BaseOptionsResearchStrategy, StrategyResearchResult


@dataclass
class BrokenWingButterflyResearchStrategy(BaseOptionsResearchStrategy):
    """Research-only broken wing butterfly for low-vol directional regimes."""

    strategy_name: str = "BrokenWingButterfly"
    min_liquidity_score: float = 0.35
    min_iv_percentile: float = 25.0
    max_iv_percentile: float = 80.0
    max_expected_move_percent: float = 3.0
    max_bid_ask_spread_percent: float = 8.0
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
        """Generate bullish call or bearish put broken wing butterflies."""

        del news_risk, iv_percentile, risk_limits
        direction = self._direction(regime)
        if direction is None:
            return ()
        option_type = "CE" if direction == "bullish" else "PE"
        contracts = contracts_by_type(option_chain_snapshot, option_type)
        lower_band, upper_band = _expected_move_band(expiry_distribution, option_chain_snapshot.spot_price)
        target = upper_band if direction == "bullish" else lower_band
        candidates: list[dict[str, Any]] = []
        for lower, middle, upper in _strike_triples(contracts):
            if direction == "bullish" and middle.strike < option_chain_snapshot.spot_price:
                continue
            if direction == "bearish" and middle.strike > option_chain_snapshot.spot_price:
                continue
            if abs(middle.strike - target) > max(abs(upper_band - lower_band), 1.0):
                continue
            lower_width = middle.strike - lower.strike
            upper_width = upper.strike - middle.strike
            if direction == "bullish" and upper_width < lower_width:
                continue
            if direction == "bearish" and lower_width < upper_width:
                continue
            candidates.append(
                {
                    "direction": direction,
                    "option_type": option_type,
                    "lower": lower,
                    "middle": middle,
                    "upper": upper,
                    "expiry_distribution": expiry_distribution,
                    "liquidity_score": liquidity_score,
                    "regime": _regime_name(regime),
                    "bid_ask_spread_percent": max(
                        _bid_ask_spread_percent(lower),
                        _bid_ask_spread_percent(middle),
                        _bid_ask_spread_percent(upper),
                    ),
                }
            )
        return candidates

    def score_candidate(self, candidate: Mapping[str, Any], **context: Any) -> StrategyResearchResult:
        """Score one broken wing butterfly candidate."""

        lower = candidate["lower"]
        middle = candidate["middle"]
        upper = candidate["upper"]
        direction = str(candidate["direction"])
        option_type = str(candidate["option_type"])
        distribution = candidate["expiry_distribution"]
        quantity = int(context.get("quantity", middle.lot_size or 1))
        kwargs = {
            "lower_strike": lower.strike,
            "body_strike": middle.strike,
            "upper_strike": upper.strike,
            "lower_premium": _buy_price(lower),
            "body_premium": _sell_price(middle),
            "upper_premium": _buy_price(upper),
            "option_type": "CALL" if option_type == "CE" else "PUT",
            "quantity": quantity,
        }
        spot = float(context.get("spot", middle.strike))
        curve = calculate_payoff_curve(
            broken_wing_butterfly_payoff,
            _price_range(distribution, spot, self.payoff_curve_points),
            **kwargs,
        )
        sample_payoffs = _distribution_payoffs(distribution, broken_wing_butterfly_payoff, **kwargs)
        max_profit = calculate_max_profit(curve)
        max_loss = calculate_max_loss(curve)
        if sample_payoffs:
            expected_value = float(np.mean(sample_payoffs))
            probability_of_profit = float(np.mean(np.asarray(sample_payoffs) > 0))
            tail_loss = abs(float(np.quantile(sample_payoffs, 0.05)))
        else:
            expected_value = 0.0
            probability_of_profit = 0.0
            tail_loss = max_loss
        risk_reward = max_profit / max_loss if max_loss > 0 else 0.0
        confidence = _clamp(
            0.25
            + probability_of_profit * 0.30
            + float(candidate.get("liquidity_score", 0.0)) * 0.15
            + min(max(float(context.get("iv_percentile", 50.0)), 0.0), 100.0) / 800.0,
            0.0,
            1.0,
        )
        result = StrategyResearchResult(
            strategy_name=f"{self.strategy_name}:{direction}",
            legs=self._legs(candidate, quantity),
            expected_value=expected_value,
            probability_of_profit=probability_of_profit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=tuple(calculate_breakevens(curve)),
            payoff_curve=tuple(curve),
            tail_loss=tail_loss,
            risk_reward=risk_reward,
            confidence=confidence,
            reject_reason=None,
            explanation=(
                f"{direction} broken wing butterfly centered at {middle.strike:.0f}; "
                f"EV {expected_value:.2f}, POP {probability_of_profit:.1%}."
            ),
            metadata=dict(candidate),
        )
        reject_reason = self.reject_candidate(result, **context)
        if reject_reason is None:
            return result
        return _replace_reject_reason(result, reject_reason)

    def reject_candidate(self, result: StrategyResearchResult, **context: Any) -> str | None:
        """Reject broken wing candidates that fail research filters."""

        news_level = str(_read_value(context.get("news_risk"), "event_risk_level") or "").upper()
        if news_level in {"HIGH", "EXTREME"}:
            return f"News risk {news_level} blocks broken wing butterfly."
        if result.max_loss <= 0:
            return "Max loss is undefined."
        if float(result.metadata.get("liquidity_score", 0.0)) < self.min_liquidity_score:
            return "Liquidity score is too low."
        if float(result.metadata.get("bid_ask_spread_percent", 0.0)) > self.max_bid_ask_spread_percent:
            return "Bid-ask spread is too wide."
        iv_percentile = float(context.get("iv_percentile", 50.0))
        if iv_percentile < self.min_iv_percentile:
            return "IV percentile is too low."
        expected_move_percent = _read_value(context.get("expiry_distribution"), "expected_move_percent")
        if expected_move_percent is not None and float(expected_move_percent) > self.max_expected_move_percent:
            return "Expected move is too large for broken wing butterfly."
        risk_limit = _max_loss_limit(context.get("risk_limits"))
        if risk_limit is not None and result.max_loss > risk_limit:
            return "Max loss exceeds risk limit."
        if result.expected_value < 0:
            return "Expected value is negative."
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
        """Generate, score, and select the best broken wing butterfly."""

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
            expiry_distribution=expiry_distribution,
        )

    @staticmethod
    def _direction(regime: Any) -> str | None:
        name = _regime_name(regime).lower()
        if "bullish" in name and "low volatility" in name:
            return "bullish"
        if "bearish" in name and "low volatility" in name:
            return "bearish"
        return None

    @staticmethod
    def _legs(candidate: Mapping[str, Any], quantity: int) -> tuple[dict[str, Any], ...]:
        lower = candidate["lower"]
        middle = candidate["middle"]
        upper = candidate["upper"]
        direction = str(candidate["direction"])
        if direction == "bearish":
            return (
                _leg_payload(upper, "BUY", quantity, _buy_price(upper)),
                _leg_payload(middle, "SELL", quantity * 2, _sell_price(middle)),
                _leg_payload(lower, "BUY", quantity, _buy_price(lower)),
            )
        return (
            _leg_payload(lower, "BUY", quantity, _buy_price(lower)),
            _leg_payload(middle, "SELL", quantity * 2, _sell_price(middle)),
            _leg_payload(upper, "BUY", quantity, _buy_price(upper)),
        )


def _strike_triples(contracts: Sequence[OptionContract]) -> list[tuple[OptionContract, OptionContract, OptionContract]]:
    sorted_contracts = sorted(contracts, key=lambda contract: contract.strike)
    triples: list[tuple[OptionContract, OptionContract, OptionContract]] = []
    for lower_index in range(len(sorted_contracts) - 2):
        for middle_index in range(lower_index + 1, len(sorted_contracts) - 1):
            for upper_index in range(middle_index + 1, len(sorted_contracts)):
                triples.append(
                    (
                        sorted_contracts[lower_index],
                        sorted_contracts[middle_index],
                        sorted_contracts[upper_index],
                    )
                )
    return triples
