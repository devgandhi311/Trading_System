"""Tests for expected-move iron condor research strategy."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from models import OptionChainSnapshot, OptionContract, RegimeResult, RegimeType
from quant.distribution.expiry_distribution import ExpiryDistributionResult
from strategies import ExpectedMoveIronCondorResearchStrategy


def _chain() -> OptionChainSnapshot:
    expiry = date.today() + timedelta(days=7)
    contracts: list[OptionContract] = []
    for strike, call, put in (
        (25500.0, 530.0, 8.0),
        (25600.0, 430.0, 16.0),
        (25700.0, 335.0, 28.0),
        (25800.0, 245.0, 45.0),
        (25900.0, 170.0, 75.0),
        (26000.0, 115.0, 115.0),
        (26100.0, 75.0, 170.0),
        (26200.0, 45.0, 245.0),
        (26300.0, 28.0, 335.0),
        (26400.0, 16.0, 430.0),
        (26500.0, 8.0, 530.0),
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
    samples = tuple(float(value) for value in np.linspace(25750.0, 26250.0, 1001))
    return ExpiryDistributionResult(
        expected_expiry_price=26000.0,
        expected_move_points=200.0,
        expected_move_percent=200.0 / 26000.0 * 100.0,
        bounds={0.68: (25800.0, 26200.0), 0.80: (25750.0, 26250.0), 0.90: (25700.0, 26300.0), 0.95: (25650.0, 26350.0)},
        tail_risk_score=0.05,
        samples=samples,
    )


def _regime(name: str = "Sideways high volatility") -> RegimeResult:
    return RegimeResult(
        symbol="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 30),
        regime=RegimeType.RANGE_BOUND,
        confidence=0.8,
        lookback_bars=100,
        regime_name=name,
        allowed_to_trade=True,
    )


def test_expected_move_iron_condor_selects_defined_risk_candidate() -> None:
    strategy = ExpectedMoveIronCondorResearchStrategy(max_bid_ask_spread_percent=15.0)

    result = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime(),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=70.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 50_000.0},
    )

    assert result is not None
    assert result.reject_reason is None
    assert len(result.legs) == 4
    assert result.probability_of_profit >= 0.65
    assert result.expected_value >= 0
    assert result.max_loss > 0
    assert result.payoff_curve


def test_expected_move_iron_condor_rejects_news_and_trend() -> None:
    strategy = ExpectedMoveIronCondorResearchStrategy(max_bid_ask_spread_percent=15.0)

    news = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime(),
        news_risk={"event_risk_level": "HIGH"},
        iv_percentile=70.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 50_000.0},
    )
    trend = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime("Bullish high volatility"),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=70.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 50_000.0},
    )

    assert news is not None
    assert "News risk HIGH" in str(news.reject_reason)
    assert trend is not None
    assert trend.reject_reason == "Strong trending regime blocks iron condor."


def test_expected_move_iron_condor_rejects_low_liquidity() -> None:
    strategy = ExpectedMoveIronCondorResearchStrategy(max_bid_ask_spread_percent=15.0)

    result = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime(),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=70.0,
        liquidity_score=0.1,
        risk_limits={"max_loss_amount": 50_000.0},
    )

    assert result is not None
    assert result.reject_reason == "Liquidity score is too low."
