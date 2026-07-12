"""Tests for performance attribution."""

from __future__ import annotations

from quant.attribution.performance_attribution import PerformanceAttribution


def test_performance_attribution_groups_and_saves(tmp_path) -> None:
    trades = [
        {"date": "2026-01-05", "strategy": "A", "regime": "Sideways", "volatility_state": "normal", "news_risk": "LOW", "days_to_expiry": 7, "gross_pnl": 100.0, "net_pnl": 80.0, "transaction_cost": 15.0, "slippage": 5.0},
        {"date": "2026-01-06", "strategy": "B", "regime": "Trending", "volatility_state": "high", "news_risk": "MEDIUM", "days_to_expiry": 3, "gross_pnl": -50.0, "net_pnl": -70.0, "transaction_cost": 15.0, "slippage": 5.0},
    ]

    attribution = PerformanceAttribution()
    result = attribution.analyze(trades)
    path = attribution.save(result, tmp_path / "attribution.xlsx")

    assert path.exists()
    assert float(result.pnl_by_strategy.loc[result.pnl_by_strategy["strategy"] == "A", "net_pnl"].iloc[0]) == 80.0
    assert result.cost_drag == 30.0
    assert result.slippage_drag == 10.0
    assert result.best_conditions
