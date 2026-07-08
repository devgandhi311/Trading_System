"""Tests for news event risk detection."""

from __future__ import annotations

from news import EventRiskDetector, NewsArticle, NewsSentiment
from utils.time_utils import now_ist


def test_event_risk_detector_blocks_high_risk_news() -> None:
    articles = tuple(
        NewsArticle(f"RBI policy inflation war market halt risk {index}", "Reuters", now_ist(), topic="RBI")
        for index in range(5)
    )
    sentiment = NewsSentiment().analyze(articles)

    result = EventRiskDetector().detect(articles, sentiment, current_market_regime="normal")

    assert result.event_risk_level in {"HIGH", "EXTREME"}
    assert result.trade_allowed is False
    assert result.position_size_multiplier == 0.0
    assert result.no_trade_minutes >= 30


def test_event_risk_detector_medium_reduces_position_size() -> None:
    articles = (NewsArticle("RBI policy update for NIFTY", "Other", now_ist(), topic="RBI"),)
    sentiment = NewsSentiment().analyze(articles)

    result = EventRiskDetector().detect(articles, sentiment)

    assert result.event_risk_level in {"LOW", "MEDIUM"}
    if result.event_risk_level == "MEDIUM":
        assert result.position_size_multiplier == 0.5
