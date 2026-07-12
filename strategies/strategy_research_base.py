"""Base abstractions for options strategy research only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class StrategyResearchResult:
    """Research-only strategy candidate evaluation."""

    strategy_name: str
    legs: tuple[dict[str, Any], ...]
    expected_value: float
    probability_of_profit: float
    max_profit: float
    max_loss: float
    breakevens: tuple[float, ...]
    payoff_curve: tuple[dict[str, float], ...]
    tail_loss: float
    risk_reward: float
    confidence: float
    reject_reason: str | None
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseOptionsResearchStrategy:
    """Base class for non-executing options research strategies."""

    strategy_name = "base_options_research_strategy"

    def generate_candidates(self, *args: Any, **kwargs: Any) -> Sequence[Mapping[str, Any]]:
        """Generate raw research candidates."""

        raise NotImplementedError

    def score_candidate(self, candidate: Mapping[str, Any], **context: Any) -> StrategyResearchResult:
        """Score one raw candidate."""

        raise NotImplementedError

    def reject_candidate(self, result: StrategyResearchResult, **context: Any) -> str | None:
        """Return a rejection reason for a scored candidate, if any."""

        del context
        return result.reject_reason

    def select_best_candidate(
        self,
        candidates: Sequence[Mapping[str, Any]],
        **context: Any,
    ) -> StrategyResearchResult | None:
        """Score candidates and return the best non-rejected result."""

        scored = [self.score_candidate(candidate, **context) for candidate in candidates]
        accepted = [result for result in scored if self.reject_candidate(result, **context) is None]
        if not accepted:
            return max(scored, key=lambda item: (item.confidence, item.expected_value), default=None)
        return max(
            accepted,
            key=lambda item: (
                item.expected_value,
                item.probability_of_profit,
                item.confidence,
                item.risk_reward,
            ),
        )

    def explain_selection(self, result: StrategyResearchResult | None) -> str:
        """Explain the selected candidate."""

        if result is None:
            return "No research candidate was generated."
        if result.reject_reason:
            return f"{result.strategy_name} rejected: {result.reject_reason}. {result.explanation}"
        return f"{result.strategy_name} selected. {result.explanation}"
