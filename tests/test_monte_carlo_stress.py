"""Tests for Monte Carlo stress testing."""

from __future__ import annotations

import numpy as np

from quant.distribution.expiry_distribution import ExpiryDistributionResult
from quant.stress import MonteCarloStressTester
from strategies.strategy_research_base import StrategyResearchResult


def _candidate() -> StrategyResearchResult:
    curve = tuple(
        {"expiry_price": float(price), "payoff": float(500.0 - abs(price - 26000.0))}
        for price in np.linspace(25500.0, 26500.0, 51)
    )
    return StrategyResearchResult("demo", (), 100.0, 0.6, 500.0, 500.0, (), curve, 200.0, 1.0, 0.7, None, "demo")


def _distribution() -> ExpiryDistributionResult:
    return ExpiryDistributionResult(
        26000.0,
        200.0,
        200.0 / 26000.0 * 100.0,
        {0.68: (25800.0, 26200.0), 0.80: (25700.0, 26300.0), 0.90: (25600.0, 26400.0), 0.95: (25500.0, 26500.0)},
        0.05,
    )


def test_monte_carlo_stress_outputs_scenario_table() -> None:
    result = MonteCarloStressTester().run(
        selected_strategy_candidate=_candidate(),
        expiry_distribution=_distribution(),
        news_event_shock="HIGH",
        max_loss_limit=600.0,
    )

    assert result.scenario_table
    assert result.worst_case_loss >= 0
    assert 0.0 <= result.probability_of_loss <= 1.0
    assert result.pass_fail in {"PASS", "FAIL"}
