"""Tests for portfolio risk optimizer."""

from __future__ import annotations

from strategies.strategy_research_base import StrategyResearchResult
from quant.optimization import PortfolioRiskOptimizer


def _candidate(max_loss: float = 1000.0, expected_value: float = 100.0) -> StrategyResearchResult:
    return StrategyResearchResult("demo", (), expected_value, 0.65, 500.0, max_loss, (), (), 200.0, 0.5, 0.7, None, "demo")


def test_portfolio_optimizer_sizes_fixed_fractional() -> None:
    rec = PortfolioRiskOptimizer(max_var_percent=5.0, max_cvar_percent=6.0).recommend_size(
        candidate_strategy=_candidate(),
        capital=1_000_000.0,
        max_risk_per_trade=10_000.0,
        max_daily_risk=20_000.0,
        var=1_000.0,
        cvar=2_000.0,
        correlation_estimate=0.2,
        news_risk={"event_risk_level": "LOW"},
    )

    assert rec.quantity >= 1
    assert rec.risk_amount > 0
    assert rec.capital_at_risk_percent > 0


def test_portfolio_optimizer_rejects_news_and_var() -> None:
    optimizer = PortfolioRiskOptimizer(max_var_percent=0.1, max_cvar_percent=0.2)
    news = optimizer.recommend_size(
        candidate_strategy=_candidate(),
        capital=100_000.0,
        max_risk_per_trade=5_000.0,
        max_daily_risk=5_000.0,
        news_risk={"event_risk_level": "HIGH"},
    )
    var = optimizer.recommend_size(
        candidate_strategy=_candidate(),
        capital=100_000.0,
        max_risk_per_trade=5_000.0,
        max_daily_risk=5_000.0,
        var=500.0,
        cvar=500.0,
        news_risk={"event_risk_level": "LOW"},
    )

    assert news.quantity == 0
    assert news.reason == "News risk too high."
    assert var.quantity == 0
    assert var.reason == "VaR limit exceeded."
