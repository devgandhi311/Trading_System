"""Tests for news fetcher fallback and cleaning."""

from __future__ import annotations

from news import NewsArticle, NewsCleaner, NewsFetcher
from utils.time_utils import now_ist
from utils.exceptions import BrokerError


class FailingProvider:
    def fetch_market_news(self):
        raise BrokerError("rate limit")


class WorkingProvider:
    def fetch_market_news(self):
        return (NewsArticle("RBI policy supports NIFTY", "Reuters", now_ist(), provider="test"),)

    def fetch_company_news(self, symbol):
        return ()

    def fetch_economic_calendar(self):
        return ()


def test_news_fetcher_uses_backup_provider() -> None:
    fetcher = NewsFetcher(providers={"finnhub": FailingProvider(), "alpha_vantage": WorkingProvider(), "marketaux": WorkingProvider()})

    articles = fetcher.fetch_all(use_cache=False)

    assert len(articles) == 1
    assert "RBI" in articles[0].headline


def test_news_cleaner_deduplicates_and_tags() -> None:
    articles = (
        NewsArticle("RBI policy and inflation impact NIFTY", "Reuters.com", now_ist()),
        NewsArticle("RBI policy and inflation impact NIFTY", "Reuters", now_ist()),
        NewsArticle("Celebrity unrelated item", "Other", now_ist()),
    )

    cleaned = NewsCleaner().clean(articles)

    assert len(cleaned) == 1
    assert cleaned[0].source == "Reuters"
    assert cleaned[0].topic == "RBI"
