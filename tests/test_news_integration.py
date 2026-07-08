"""Tests for news integration hooks."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from features import FeaturePipeline
from models import OrderSide, OrderType, RegimeType, StrategyLeg, StrategyPlan, StrategyType
from ml import PredictionEngine
from news import EventRiskResult, SentimentResult
from risk import PositionSizer, RiskManager
from utils.time_utils import KOLKATA_TZ


MARKET_TIME = datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ)


def _plan(metadata=None) -> StrategyPlan:
    return StrategyPlan(
        strategy_id="news-risk",
        strategy_type=StrategyType.CREDIT_SPREAD,
        symbol="NIFTY",
        created_at=MARKET_TIME,
        legs=(StrategyLeg("NIFTY26000CE", OrderSide.SELL, 75, OrderType.LIMIT, limit_price=100.0),),
        confidence=0.8,
        rationale="test",
        max_loss=2000.0,
        metadata=metadata or {},
    )


def test_feature_pipeline_adds_news_columns() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=25, freq="B"),
            "open": range(100, 125),
            "high": range(101, 126),
            "low": range(99, 124),
            "close": range(100, 125),
            "volume": [1000] * 25,
        }
    )

    result = FeaturePipeline(mode="BACKTEST").generate_features(
        data,
        cache=False,
        latest_news_sentiment_score=-0.3,
        latest_event_risk_score=0.6,
    )

    assert "news_sentiment_score" in result.features.columns
    assert result.features["event_risk_score"].iloc[-1] == 0.6


def test_risk_manager_rejects_high_news_risk() -> None:
    decision = RiskManager().validate(
        _plan(),
        account_context={"daily_pnl": 0, "trades_today": 0, "open_positions": 0, "consecutive_losses": 0, "margin_used": 0},
        market_context={"event_risk_level": "HIGH", "bid_ask_spread_percent": 0.1, "slippage_percent": 0.1},
        timestamp=MARKET_TIME,
    )

    assert decision.approved is False
    assert "News event risk HIGH" in decision.rejection_reason


def test_position_sizer_reduces_for_medium_news_risk() -> None:
    result = PositionSizer(capital=1_000_000, max_loss_per_trade_percent=1.0).size(
        _plan(),
        event_risk_level="MEDIUM",
        position_size_multiplier=0.5,
    )

    assert result.allowed_quantity == 37


def test_prediction_engine_accepts_news_context() -> None:
    sentiment = SentimentResult(-0.5, "NEGATIVE", 0.8, (), (), {}, "test")
    event_risk = EventRiskResult(0.7, "HIGH", False, 0.0, 30, "risk", ())
    prediction = PredictionEngine().predict(
        {"symbol": "NIFTY", "close": 25000, "daily_return": 0.01, "liquidity_score": 0.8},
        RegimeType.TRENDING,
        news_sentiment=sentiment,
        news_event_risk=event_risk,
    )

    assert prediction.metadata["news_sentiment_score"] == -0.5
    assert prediction.metadata["event_risk_level"] == "HIGH"
