"""Tests for broken wing butterfly research strategy."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from models import OptionChainSnapshot, OptionContract, RegimeResult, RegimeType
from quant.distribution.expiry_distribution import ExpiryDistributionResult
from strategies import BrokenWingButterflyResearchStrategy


def _chain() -> OptionChainSnapshot:
    expiry = date.today() + timedelta(days=14)
    contracts: list[OptionContract] = []
    for strike, call, put in (
        (25600.0, 470.0, 35.0),
        (25700.0, 380.0, 48.0),
        (25800.0, 295.0, 70.0),
        (25900.0, 220.0, 100.0),
        (26000.0, 160.0, 150.0),
        (26100.0, 110.0, 215.0),
        (26200.0, 72.0, 292.0),
        (26300.0, 46.0, 378.0),
        (26400.0, 30.0, 470.0),
    ):
        contracts.extend(
            (
                OptionContract(
                    symbol=f"NIFTY{int(strike)}CE",
                    underlying="NIFTY",
                    expiry=expiry,
                    strike=strike,
                    option_type="CE",
                    lot_size=75,
                    bid=call - 0.5,
                    ask=call + 0.5,
                    last_price=call,
                ),
                OptionContract(
                    symbol=f"NIFTY{int(strike)}PE",
                    underlying="NIFTY",
                    expiry=expiry,
                    strike=strike,
                    option_type="PE",
                    lot_size=75,
                    bid=put - 0.5,
                    ask=put + 0.5,
                    last_price=put,
                ),
            )
        )
    return OptionChainSnapshot("NIFTY", datetime(2026, 1, 1, 9, 30), 26000.0, expiry, tuple(contracts))


def _distribution() -> ExpiryDistributionResult:
    samples = tuple(float(value) for value in np.linspace(25900.0, 26300.0, 801))
    return ExpiryDistributionResult(
        expected_expiry_price=26100.0,
        expected_move_points=250.0,
        expected_move_percent=250.0 / 26000.0 * 100.0,
        bounds={0.68: (25850.0, 26350.0), 0.80: (25800.0, 26400.0), 0.90: (25750.0, 26450.0), 0.95: (25700.0, 26500.0)},
        tail_risk_score=0.08,
        samples=samples,
    )


def _regime(name: str = "Bullish low volatility") -> RegimeResult:
    return RegimeResult(
        symbol="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 30),
        regime=RegimeType.LOW_VOLATILITY,
        confidence=0.8,
        lookback_bars=100,
        regime_name=name,
        allowed_to_trade=True,
    )


def test_broken_wing_butterfly_selects_bullish_candidate() -> None:
    strategy = BrokenWingButterflyResearchStrategy(max_bid_ask_spread_percent=10.0)

    result = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime("Bullish low volatility"),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=45.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 100_000.0},
    )

    assert result is not None
    assert result.reject_reason is None
    assert result.strategy_name.endswith("bullish")
    assert len(result.legs) == 3
    assert result.max_loss > 0
    assert result.expected_value >= 0
    assert result.payoff_curve


def test_broken_wing_butterfly_supports_bearish_low_vol_regime() -> None:
    bearish_distribution = ExpiryDistributionResult(
        expected_expiry_price=25900.0,
        expected_move_points=250.0,
        expected_move_percent=250.0 / 26000.0 * 100.0,
        bounds={0.68: (25650.0, 26150.0), 0.80: (25600.0, 26200.0), 0.90: (25550.0, 26250.0), 0.95: (25500.0, 26300.0)},
        tail_risk_score=0.08,
        samples=tuple(float(value) for value in np.linspace(25700.0, 26100.0, 801)),
    )
    strategy = BrokenWingButterflyResearchStrategy(max_bid_ask_spread_percent=10.0)

    result = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=bearish_distribution,
        regime=_regime("Bearish low volatility"),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=45.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 100_000.0},
    )

    assert result is not None
    assert result.strategy_name.endswith("bearish")
    assert result.legs[0]["option_type"] == "PE"


def test_broken_wing_butterfly_rejects_news_and_low_iv() -> None:
    strategy = BrokenWingButterflyResearchStrategy(max_bid_ask_spread_percent=10.0)

    news = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime(),
        news_risk={"event_risk_level": "EXTREME"},
        iv_percentile=45.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 100_000.0},
    )
    low_iv = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime(),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=10.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 100_000.0},
    )

    assert news is not None
    assert "News risk EXTREME" in str(news.reject_reason)
    assert low_iv is not None
    assert low_iv.reject_reason == "IV percentile is too low."


def test_broken_wing_butterfly_skips_non_directional_regime() -> None:
    result = BrokenWingButterflyResearchStrategy().analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime("Sideways low volatility"),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=45.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 100_000.0},
    )

    assert result is None
