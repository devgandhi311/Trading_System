"""Tests for prediction explainability and strategy selection."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from ml.explainability import ExplainabilityEngine
from ml.prediction_engine import PredictionEngine
from models import OptionChainSnapshot, OptionContract, RegimeResult, RegimeType, StrategyType
from strategies import StrategySelector


def _option_chain() -> OptionChainSnapshot:
    expiry = date.today() + timedelta(days=7)
    contracts = (
        OptionContract(
            symbol="NIFTY26000CE",
            underlying="NIFTY",
            expiry=expiry,
            strike=26000.0,
            option_type="CE",
            lot_size=75,
            bid=80.0,
            ask=82.0,
            last_price=81.0,
            implied_volatility=18.0,
        ),
        OptionContract(
            symbol="NIFTY26000PE",
            underlying="NIFTY",
            expiry=expiry,
            strike=26000.0,
            option_type="PE",
            lot_size=75,
            bid=76.0,
            ask=78.0,
            last_price=77.0,
            implied_volatility=19.0,
        ),
        OptionContract(
            symbol="NIFTY26100CE",
            underlying="NIFTY",
            expiry=expiry,
            strike=26100.0,
            option_type="CE",
            lot_size=75,
            bid=46.0,
            ask=48.0,
            last_price=47.0,
            implied_volatility=20.0,
        ),
        OptionContract(
            symbol="NIFTY25900PE",
            underlying="NIFTY",
            expiry=expiry,
            strike=25900.0,
            option_type="PE",
            lot_size=75,
            bid=44.0,
            ask=46.0,
            last_price=45.0,
            implied_volatility=20.0,
        ),
    )
    return OptionChainSnapshot(
        underlying="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 20),
        spot_price=26025.0,
        expiry=expiry,
        contracts=contracts,
    )


def test_prediction_engine_returns_rich_prediction() -> None:
    features = {
        "symbol": "NIFTY",
        "close": 26025.0,
        "daily_return": 0.012,
        "rolling_mean_return": 0.01,
        "rolling_volatility": 0.08,
        "volatility_percentile": 72.0,
        "liquidity_score": 0.8,
        "iv_percentile": 68.0,
        "expected_move_from_iv": 1.2,
    }
    regime = RegimeResult(
        symbol="NIFTY",
        timestamp=datetime(2026, 1, 1, 9, 20),
        regime=RegimeType.TRENDING,
        confidence=0.8,
        confidence_score=0.8,
        lookback_bars=60,
        allowed_to_trade=True,
        regime_name="Bullish high volatility",
    )

    result = PredictionEngine().predict(features, regime, _option_chain())

    assert result.underlying == "NIFTY"
    assert result.bullish_probability > result.bearish_probability
    assert result.expected_range_low is not None
    assert result.expected_range_high is not None
    assert result.volatility_state == "expanding"
    assert result.trade_allowed is True
    assert result.recommended_strategy_type in {item.value for item in StrategyType}
    assert result.metadata["explainability"]["top_features"]


def test_explainability_warning_flags() -> None:
    explanation = ExplainabilityEngine().build(
        {"bid_ask_spread_percent": 1.5, "liquidity_score": 0.2, "abnormal_candle": True},
        confidence=0.4,
        bullish_probability=0.3,
        bearish_probability=0.3,
        sideways_probability=0.4,
        breakout_failure_probability=0.8,
        trade_allowed=False,
        strategy_type=None,
        regime_name="Event-risk no-trade",
        expected_move_percent=1.0,
    )

    assert "trade_not_allowed_by_regime_or_config" in explanation.warning_flags
    assert "wide_bid_ask_spread" in explanation.warning_flags
    assert "No strategy suggested" in explanation.strategy_suggestion_reason


def test_strategy_selector_returns_valid_plan_for_allowed_prediction() -> None:
    prediction = PredictionEngine().predict(
        {
            "symbol": "NIFTY",
            "close": 26025.0,
            "daily_return": 0.012,
            "rolling_mean_return": 0.01,
            "rolling_volatility": 0.08,
            "volatility_percentile": 72.0,
            "liquidity_score": 0.85,
            "iv_percentile": 75.0,
            "expected_move_from_iv": 1.4,
        },
        RegimeResult(
            symbol="NIFTY",
            timestamp=datetime(2026, 1, 1, 9, 20),
            regime=RegimeType.TRENDING,
            confidence=0.85,
            confidence_score=0.85,
            lookback_bars=60,
            allowed_to_trade=True,
            regime_name="Bullish high volatility",
        ),
        _option_chain(),
    )

    plan = StrategySelector().select(
        prediction,
        _option_chain(),
        features={"liquidity_score": 0.85, "iv_percentile": 75.0},
        risk_context={"estimated_loss": 1000.0, "max_loss_allowed": 10000.0},
    )

    assert plan.metadata["valid"] is True
    assert plan.strategy_type in {
        StrategyType.MOMENTUM_BREAKOUT,
        StrategyType.VOLATILITY_SCALPING,
        StrategyType.MEAN_REVERSION,
        StrategyType.EXPECTED_MOVE_CREDIT_SPREAD,
    }
    assert plan.legs
    assert plan.metadata["advisory_only"] is True


def test_strategy_selector_blocks_low_liquidity() -> None:
    prediction = PredictionEngine().predict(
        {
            "symbol": "NIFTY",
            "close": 26025.0,
            "daily_return": 0.02,
            "rolling_mean_return": 0.02,
            "rolling_volatility": 0.02,
            "volatility_percentile": 72.0,
            "liquidity_score": 0.7,
            "expected_move_from_iv": 1.2,
        },
        RegimeResult(
            symbol="NIFTY",
            timestamp=datetime(2026, 1, 1, 9, 20),
            regime=RegimeType.TRENDING,
            confidence=0.9,
            confidence_score=0.9,
            lookback_bars=60,
            allowed_to_trade=True,
        ),
        _option_chain(),
    )

    plan = StrategySelector().select(
        prediction,
        _option_chain(),
        features={"liquidity_score": 0.1},
    )

    assert plan.metadata["valid"] is False
    assert plan.legs == ()
    assert "Liquidity" in plan.rationale
