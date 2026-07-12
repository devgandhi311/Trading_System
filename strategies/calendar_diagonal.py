"""Calendar and diagonal spread research placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.market_models import OptionChainSnapshot
from strategies.strategy_research_base import BaseOptionsResearchStrategy, StrategyResearchResult


@dataclass
class CalendarSpreadStrategy(BaseOptionsResearchStrategy):
    """Placeholder interface for future calendar spread research."""

    strategy_name: str = "CalendarSpread"

    def generate_candidates(self, **kwargs: Any) -> tuple[dict[str, Any], ...]:
        return (dict(kwargs),)

    def score_candidate(self, candidate: dict[str, Any], **context: Any) -> StrategyResearchResult:
        del context
        near = candidate.get("near_expiry_option_chain")
        far = candidate.get("far_expiry_option_chain")
        reason = _insufficient_reason(near, far, candidate.get("iv_term_structure"))
        return StrategyResearchResult(
            strategy_name=self.strategy_name,
            legs=(),
            expected_value=0.0,
            probability_of_profit=0.0,
            max_profit=0.0,
            max_loss=0.0,
            breakevens=(),
            payoff_curve=(),
            tail_loss=0.0,
            risk_reward=0.0,
            confidence=0.0,
            reject_reason=reason or "Calendar spread valuation is not implemented yet.",
            explanation="Calendar spread placeholder requires term-structure repricing before use.",
            metadata=candidate,
        )

    def analyze(self, **kwargs: Any) -> StrategyResearchResult:
        return self.score_candidate(dict(kwargs))


@dataclass
class DiagonalSpreadStrategy(CalendarSpreadStrategy):
    """Placeholder interface for future diagonal spread research."""

    strategy_name: str = "DiagonalSpread"

    def score_candidate(self, candidate: dict[str, Any], **context: Any) -> StrategyResearchResult:
        result = super().score_candidate(candidate, **context)
        return StrategyResearchResult(
            strategy_name=self.strategy_name,
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
            reject_reason=result.reject_reason or "Diagonal spread valuation is not implemented yet.",
            explanation="Diagonal spread placeholder requires strike and term-structure repricing before use.",
            metadata=result.metadata,
        )


def _insufficient_reason(
    near_expiry_option_chain: OptionChainSnapshot | None,
    far_expiry_option_chain: OptionChainSnapshot | None,
    iv_term_structure: Any | None,
) -> str | None:
    if near_expiry_option_chain is None:
        return "Near-expiry option chain is required."
    if far_expiry_option_chain is None:
        return "Far-expiry option chain is required."
    if not near_expiry_option_chain.contracts or not far_expiry_option_chain.contracts:
        return "Near and far option chains must include contracts."
    if iv_term_structure is None:
        return "IV term structure is required."
    return None
