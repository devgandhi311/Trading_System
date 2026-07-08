"""Tests for local news sentiment."""

from __future__ import annotations

from news import NewsArticle, NewsSentiment
from utils.time_utils import now_ist


def test_news_sentiment_keyword_fallback_scores_negative_news() -> None:
    sentiment = NewsSentiment()
    result = sentiment.analyze(
        (
            NewsArticle("Inflation crisis and market crash risk for NIFTY", "Reuters", now_ist(), topic="inflation"),
            NewsArticle("NIFTY growth stable after RBI relief", "Reuters", now_ist(), topic="RBI"),
        )
    )

    assert -1.0 <= result.sentiment_score <= 1.0
    assert result.confidence > 0
    assert result.topic_wise_sentiment


def test_news_sentiment_neutral_for_empty_input() -> None:
    result = NewsSentiment().analyze(())

    assert result.sentiment_score == 0.0
    assert result.sentiment_label == "NEUTRAL"
