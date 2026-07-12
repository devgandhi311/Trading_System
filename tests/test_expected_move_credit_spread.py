"""Tests for expected-move credit spread research strategy."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from models import OptionChainSnapshot, OptionContract, RegimeResult, RegimeType
from quant.distribution.expiry_distribution import ExpiryDistributionResult
from strategies import BaseOptionsResearchStrategy, ExpectedMoveCreditSpreadResearchStrategy, StrategyResearchResult


def _chain() -> OptionChainSnapshot:
    expiry = date.today() + timedelta(days=7)
    contracts: list[OptionContract] = []
    for strike, call, put in (
        (25500.0, 620.0, 12.0),
        (25600.0, 520.0, 20.0),
        (25700.0, 420.0, 32.0),
        (25800.0, 330.0, 48.0),
        (25900.0, 245.0, 75.0),
        (26000.0, 170.0, 120.0),
        (26100.0, 120.0, 175.0),
        (26200.0, 78.0, 245.0),
        (26300.0, 50.0, 330.0),
        (26400.0, 32.0, 425.0),
        (26500.0, 20.0, 530.0),
    ):
        contracts.append(
            OptionContract(
                symbol=f"NIFTY{int(strike)}CE",
                underlying="NIFTY",
                expiry=expiry,
                strike=strike,
                option_type="CE",
                lot_size=75,
                bid=call - 1.0,
                ask=call + 1.0,
                last_price=call,
            )
        )
        contracts.append(
            OptionContract(
                symbol=f"NIFTY{int(strike)}PE",
                underlying="NIFTY",
                expiry=expiry,
                strike=strike,
                option_type="PE",
                lot_size=75,
                bid=put - 1.0,
                ask=put + 1.0,
                last_price=put,
            )
        )
    return OptionChainSnapshot(
        underlying="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 30),
        spot_price=26000.0,
        expiry=expiry,
        contracts=tuple(contracts),
    )


def _distribution() -> ExpiryDistributionResult:
    samples = tuple(float(value) for value in np.linspace(25750.0, 26450.0, 701))
    return ExpiryDistributionResult(
        expected_expiry_price=26000.0,
        expected_move_points=200.0,
        expected_move_percent=200.0 / 26000.0 * 100.0,
        bounds={
            0.68: (25800.0, 26200.0),
            0.80: (25750.0, 26250.0),
            0.90: (25650.0, 26350.0),
            0.95: (25550.0, 26450.0),
        },
        tail_risk_score=0.08,
        samples=samples,
    )


def _regime(name: str = "Bullish low volatility") -> RegimeResult:
    return RegimeResult(
        symbol="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 30),
        regime=RegimeType.TRENDING,
        confidence=0.8,
        lookback_bars=100,
        regime_name=name,
        allowed_to_trade=True,
    )


def test_research_strategy_generates_and_selects_bull_put_candidate() -> None:
    strategy = ExpectedMoveCreditSpreadResearchStrategy(max_bid_ask_spread_percent=12.0)

    result = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime("Bullish low volatility"),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=65.0,
        liquidity_score=0.8,
        risk_limits={"capital": 1_000_000.0, "max_loss_per_trade_percent": 2.0},
    )

    assert result is not None
    assert result.reject_reason is None
    assert result.strategy_name.endswith("bull_put")
    assert len(result.legs) == 2
    assert result.expected_value >= 0
    assert result.probability_of_profit >= 0.65
    assert result.max_loss > 0
    assert result.risk_reward > 0
    assert result.payoff_curve


def test_sideways_regime_considers_both_sides() -> None:
    strategy = ExpectedMoveCreditSpreadResearchStrategy(max_bid_ask_spread_percent=12.0)

    candidates = strategy.generate_candidates(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime("Sideways low volatility"),
        news_risk={"event_risk_level": "LOW"},
        iv_percentile=70.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 50_000.0},
    )

    variants = {candidate["variant"] for candidate in candidates}
    assert {"bull_put", "bear_call"}.issubset(variants)


def test_research_strategy_rejects_high_news_risk() -> None:
    strategy = ExpectedMoveCreditSpreadResearchStrategy(max_bid_ask_spread_percent=12.0)

    result = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime("Bullish low volatility"),
        news_risk={"event_risk_level": "HIGH"},
        iv_percentile=65.0,
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 50_000.0},
    )

    assert result is not None
    assert result.reject_reason is not None
    assert "News risk HIGH" in result.reject_reason


def test_research_strategy_rejects_low_liquidity_and_tight_risk_limit() -> None:
    strategy = ExpectedMoveCreditSpreadResearchStrategy(max_bid_ask_spread_percent=12.0)

    low_liquidity = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime("Bullish low volatility"),
        news_risk={"event_risk_level": "LOW"},
        liquidity_score=0.1,
        risk_limits={"max_loss_amount": 50_000.0},
    )
    tight_risk = strategy.analyze(
        option_chain_snapshot=_chain(),
        expiry_distribution=_distribution(),
        regime=_regime("Bullish low volatility"),
        news_risk={"event_risk_level": "LOW"},
        liquidity_score=0.8,
        risk_limits={"max_loss_amount": 1.0},
    )

    assert low_liquidity is not None
    assert low_liquidity.reject_reason == "Liquidity score is too low."
    assert tight_risk is not None
    assert tight_risk.reject_reason == "Max loss exceeds risk limit."


def test_base_research_strategy_explains_rejection() -> None:
    result = StrategyResearchResult(
        strategy_name="demo",
        legs=(),
        expected_value=-1.0,
        probability_of_profit=0.4,
        max_profit=1.0,
        max_loss=2.0,
        breakevens=(),
        payoff_curve=(),
        tail_loss=2.0,
        risk_reward=0.5,
        confidence=0.2,
        reject_reason="bad candidate",
        explanation="demo explanation",
    )

    assert "rejected" in BaseOptionsResearchStrategy().explain_selection(result)
