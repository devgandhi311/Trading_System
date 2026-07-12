"""Tests for research StrategySelectorV2."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np

from models import OptionChainSnapshot, OptionContract, RegimeResult, RegimeType
from quant.distribution.expiry_distribution import ExpiryDistributionResult
from quant.volatility import VolForecastResult
from strategies import StrategySelectorV2


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


def test_strategy_selector_v2_selects_candidate() -> None:
    selector = StrategySelectorV2(allow_calendar_diagonal_placeholders=True)
    prediction = SimpleNamespace(
        metadata={"iv_percentile": 80.0, "liquidity_score": 0.8, "implied_volatility": 0.30, "realized_volatility": 0.12},
        breakout_failure_probability=0.7,
    )
    regime = RegimeResult("NIFTY", datetime(2026, 1, 1), RegimeType.RANGE_BOUND, 0.8, 100, regime_name="Sideways high volatility", allowed_to_trade=True)
    vol = VolForecastResult(0.18, 18.0, "normal", 0.7, "test")

    result = selector.select(
        current_regime=regime,
        prediction_result=prediction,
        expiry_distribution=_distribution(),
        volatility_forecast=vol,
        option_chain=_chain(),
        news_risk={"event_risk_level": "LOW"},
        risk_limits={"max_loss_amount": 100_000.0},
    )

    assert result.selected_strategy != "NO_TRADE"
    assert result.selected_candidate is not None
    assert result.all_candidates_summary


def test_strategy_selector_v2_returns_no_trade_when_news_blocks() -> None:
    selector = StrategySelectorV2()
    prediction = SimpleNamespace(metadata={"iv_percentile": 80.0, "liquidity_score": 0.8}, breakout_failure_probability=0.7)
    regime = RegimeResult("NIFTY", datetime(2026, 1, 1), RegimeType.RANGE_BOUND, 0.8, 100, regime_name="Sideways high volatility", allowed_to_trade=True)

    result = selector.select(
        current_regime=regime,
        prediction_result=prediction,
        expiry_distribution=_distribution(),
        volatility_forecast=VolForecastResult(0.18, 18.0, "normal", 0.7, "test"),
        option_chain=_chain(),
        news_risk={"event_risk_level": "EXTREME"},
        risk_limits={"max_loss_amount": 100_000.0},
    )

    assert result.selected_strategy == "NO_TRADE"
    assert result.rejection_reasons
