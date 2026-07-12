"""Tests for volatility expansion research strategy."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from models import OptionChainSnapshot, OptionContract
from quant.distribution.expiry_distribution import ExpiryDistributionResult
from strategies import VolatilityExpansionResearchStrategy


def _chain() -> OptionChainSnapshot:
    expiry = date.today() + timedelta(days=7)
    contracts = []
    for strike, call, put in ((25900.0, 120.0, 80.0), (26000.0, 70.0, 70.0), (26100.0, 35.0, 105.0), (26200.0, 18.0, 150.0)):
        contracts.append(OptionContract(f"NIFTY{int(strike)}CE", "NIFTY", expiry, strike, "CE", 75, bid=call - 0.5, ask=call + 0.5, last_price=call))
        contracts.append(OptionContract(f"NIFTY{int(strike)}PE", "NIFTY", expiry, strike, "PE", 75, bid=put - 0.5, ask=put + 0.5, last_price=put))
    return OptionChainSnapshot("NIFTY", datetime(2026, 1, 1), 26000.0, expiry, tuple(contracts))


def _distribution() -> ExpiryDistributionResult:
    samples = tuple(float(value) for value in np.concatenate([np.linspace(25500.0, 25850.0, 250), np.linspace(26150.0, 26500.0, 250)]))
    return ExpiryDistributionResult(
        26000.0,
        450.0,
        450.0 / 26000.0 * 100.0,
        {0.68: (25700.0, 26300.0), 0.80: (25600.0, 26400.0), 0.90: (25500.0, 26500.0), 0.95: (25450.0, 26550.0)},
        0.12,
        samples,
    )


def test_volatility_expansion_selects_positive_ev_candidate() -> None:
    result = VolatilityExpansionResearchStrategy().analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        realized_volatility=0.10,
        forecast_volatility=0.22,
        breakout_probability=0.75,
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=55.0,
        risk_limits={"max_loss_amount": 100_000.0},
    )

    assert result is not None
    assert result.reject_reason is None
    assert result.strategy_name.startswith("VolatilityExpansion")
    assert result.expected_value >= 0


def test_volatility_expansion_rejects_extreme_news_and_high_iv() -> None:
    strategy = VolatilityExpansionResearchStrategy()

    news = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        realized_volatility=0.10,
        forecast_volatility=0.22,
        breakout_probability=0.75,
        news_risk={"event_risk_level": "EXTREME"},
        iv_percentile=55.0,
    )
    high_iv = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        realized_volatility=0.10,
        forecast_volatility=0.22,
        breakout_probability=0.75,
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=90.0,
    )

    assert news is not None
    assert news.reject_reason == "EXTREME news risk blocks volatility expansion."
    assert high_iv is not None
    assert high_iv.reject_reason == "IV is too expensive for long premium."
