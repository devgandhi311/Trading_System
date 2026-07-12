"""Tests for IV mean-reversion premium research strategy."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from models import OptionChainSnapshot, OptionContract, RegimeResult, RegimeType
from quant.distribution.expiry_distribution import ExpiryDistributionResult
from strategies import IVMeanReversionPremiumResearchStrategy


def _chain() -> OptionChainSnapshot:
    expiry = date.today() + timedelta(days=7)
    contracts = []
    for strike, call, put in ((25800.0, 60.0, 30.0), (25900.0, 40.0, 45.0), (26000.0, 25.0, 60.0), (26100.0, 16.0, 85.0), (26200.0, 10.0, 120.0)):
        contracts.append(OptionContract(f"NIFTY{int(strike)}CE", "NIFTY", expiry, strike, "CE", 75, bid=call - 0.5, ask=call + 0.5, last_price=call))
        contracts.append(OptionContract(f"NIFTY{int(strike)}PE", "NIFTY", expiry, strike, "PE", 75, bid=put - 0.5, ask=put + 0.5, last_price=put))
    return OptionChainSnapshot("NIFTY", datetime(2026, 1, 1), 26000.0, expiry, tuple(contracts))


def _distribution() -> ExpiryDistributionResult:
    return ExpiryDistributionResult(
        26000.0,
        120.0,
        120.0 / 26000.0 * 100.0,
        {0.68: (25900.0, 26100.0), 0.80: (25850.0, 26150.0), 0.90: (25800.0, 26200.0), 0.95: (25750.0, 26250.0)},
        0.05,
        tuple(float(value) for value in np.linspace(25900.0, 26100.0, 401)),
    )


def _regime() -> RegimeResult:
    return RegimeResult("NIFTY", datetime(2026, 1, 1), RegimeType.RANGE_BOUND, 0.8, 100, regime_name="Sideways high volatility", allowed_to_trade=True)


def test_iv_mean_reversion_selects_defined_risk_premium_strategy() -> None:
    result = IVMeanReversionPremiumResearchStrategy().analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime(),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=80.0,
        implied_volatility=0.30,
        forecast_volatility=0.18,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 100_000.0},
    )

    assert result is not None
    assert result.reject_reason is None
    assert result.strategy_name.startswith("IVMeanReversionPremium")
    assert result.legs


def test_iv_mean_reversion_rejects_low_spread_and_news() -> None:
    strategy = IVMeanReversionPremiumResearchStrategy()

    low_spread = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime(),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=80.0,
        implied_volatility=0.20,
        forecast_volatility=0.19,
        liquidity_score=0.8,
    )
    news = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime(),
        news_risk={"event_risk_level": "HIGH"},
        iv_percentile=80.0,
        implied_volatility=0.30,
        forecast_volatility=0.18,
        liquidity_score=0.8,
    )

    assert low_spread is not None
    assert low_spread.reject_reason == "IV minus forecast volatility spread is too small."
    assert news is not None
    assert "News risk HIGH" in str(news.reject_reason)
