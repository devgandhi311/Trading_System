"""IV mean-reversion premium-selling research strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from models.market_models import OptionChainSnapshot
from strategies.expected_move_credit_spread import ExpectedMoveCreditSpreadResearchStrategy, _read_value, _regime_name
from strategies.expected_move_iron_condor import ExpectedMoveIronCondorResearchStrategy
from strategies.strategy_research_base import BaseOptionsResearchStrategy, StrategyResearchResult


@dataclass
class IVMeanReversionPremiumResearchStrategy(BaseOptionsResearchStrategy):
    """Research-only premium selling when IV is rich versus forecast volatility."""

    strategy_name: str = "IVMeanReversionPremium"
    min_iv_percentile: float = 65.0
    min_iv_forecast_spread: float = 0.03
    allow_hedged_short_strangle: bool = False

    def generate_candidates(
        self,
        *,
        option_chain_snapshot: OptionChainSnapshot,
        expiry_distribution: Any,
        regime: Any,
        news_risk: Any | None = None,
        iv_percentile: float,
        implied_volatility: float,
        forecast_volatility: float,
        liquidity_score: float = 0.6,
        risk_limits: Any | None = None,
        major_event_nearby: bool = False,
    ) -> Sequence[Mapping[str, Any]]:
        """Generate allowed premium-selling structures."""

        del option_chain_snapshot, expiry_distribution, risk_limits
        spread = _as_decimal_vol(implied_volatility) - _as_decimal_vol(forecast_volatility)
        return (
            {
                "structure": "iron_condor",
                "regime": _regime_name(regime),
                "news_risk": news_risk,
                "iv_percentile": iv_percentile,
                "iv_forecast_spread": spread,
                "liquidity_score": liquidity_score,
                "major_event_nearby": major_event_nearby,
            },
            {
                "structure": "credit_spread",
                "regime": _regime_name(regime),
                "news_risk": news_risk,
                "iv_percentile": iv_percentile,
                "iv_forecast_spread": spread,
                "liquidity_score": liquidity_score,
                "major_event_nearby": major_event_nearby,
            },
        ) + (
            (
                {
                    "structure": "hedged_short_strangle",
                    "regime": _regime_name(regime),
                    "news_risk": news_risk,
                    "iv_percentile": iv_percentile,
                    "iv_forecast_spread": spread,
                    "liquidity_score": liquidity_score,
                    "major_event_nearby": major_event_nearby,
                },
            )
            if self.allow_hedged_short_strangle
            else ()
        )

    def score_candidate(self, candidate: Mapping[str, Any], **context: Any) -> StrategyResearchResult:
        """Delegate defined-risk structure scoring to existing research strategies."""

        structure = str(candidate["structure"])
        option_chain_snapshot = context["option_chain_snapshot"]
        expiry_distribution = context["expiry_distribution"]
        regime = context["regime"]
        risk_limits = context.get("risk_limits")
        news_risk = candidate.get("news_risk")
        iv_percentile = float(candidate["iv_percentile"])
        liquidity_score = float(candidate["liquidity_score"])

        if structure == "iron_condor":
            result = ExpectedMoveIronCondorResearchStrategy(
                allow_strong_trend=True,
                max_bid_ask_spread_percent=15.0,
            ).analyze(
                option_chain_snapshot=option_chain_snapshot,
                expiry_distribution=expiry_distribution,
                regime=regime,
                news_risk=news_risk,
                iv_percentile=iv_percentile,
                liquidity_score=liquidity_score,
                risk_limits=risk_limits,
            )
        elif structure == "credit_spread":
            result = ExpectedMoveCreditSpreadResearchStrategy(max_bid_ask_spread_percent=15.0).analyze(
                option_chain_snapshot=option_chain_snapshot,
                expiry_distribution=expiry_distribution,
                regime=regime,
                news_risk=news_risk,
                iv_percentile=iv_percentile,
                liquidity_score=liquidity_score,
                risk_limits=risk_limits,
            )
        else:
            result = None

        if result is None:
            result = _empty_result(self.strategy_name, "Insufficient data for candidate.")
        renamed = _rename(result, f"{self.strategy_name}:{structure}", dict(candidate))
        reject_reason = self.reject_candidate(renamed, **context)
        return _with_reject_reason(renamed, reject_reason)

    def reject_candidate(self, result: StrategyResearchResult, **context: Any) -> str | None:
        """Reject when IV premium is not rich enough or event risk is present."""

        metadata = result.metadata
        news_level = str(_read_value(metadata.get("news_risk") or context.get("news_risk"), "event_risk_level") or "LOW").upper()
        if news_level != "LOW":
            return f"News risk {news_level} blocks IV mean reversion."
        if bool(metadata.get("major_event_nearby") or context.get("major_event_nearby", False)):
            return "Major event nearby blocks premium selling."
        if float(metadata.get("iv_percentile", 0.0)) < self.min_iv_percentile:
            return "IV percentile is not high enough."
        if float(metadata.get("iv_forecast_spread", 0.0)) < self.min_iv_forecast_spread:
            return "IV minus forecast volatility spread is too small."
        regime_name = str(metadata.get("regime", "")).lower()
        if not any(word in regime_name for word in ("sideways", "range", "bullish", "bearish", "trend")):
            return "Regime is not compatible with controlled premium selling."
        if result.reject_reason:
            return result.reject_reason
        return None

    def analyze(self, **kwargs: Any) -> StrategyResearchResult | None:
        """Generate, score, and select best IV mean-reversion candidate."""

        candidates = self.generate_candidates(**kwargs)
        return self.select_best_candidate(candidates, **kwargs)


def _as_decimal_vol(value: float) -> float:
    vol = abs(float(value))
    return vol / 100.0 if vol > 1.0 else vol


def _empty_result(strategy_name: str, reason: str) -> StrategyResearchResult:
    return StrategyResearchResult(strategy_name, (), 0.0, 0.0, 0.0, 0.0, (), (), 0.0, 0.0, 0.0, reason, reason)


def _rename(result: StrategyResearchResult, strategy_name: str, metadata: dict[str, Any]) -> StrategyResearchResult:
    merged = dict(result.metadata)
    merged.update(metadata)
    return StrategyResearchResult(
        strategy_name=strategy_name,
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
        reject_reason=result.reject_reason,
        explanation=result.explanation,
        metadata=merged,
    )


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
