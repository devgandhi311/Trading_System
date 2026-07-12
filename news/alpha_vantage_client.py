"""Alpha Vantage news backup adapter."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    requests = None  # type: ignore[assignment]

from news.news_config import NewsArticle, article_from_mapping
from utils.exceptions import BrokerError
from utils.env_loader import load_project_env

LOGGER = logging.getLogger(__name__)


class AlphaVantageClient:
    """Alpha Vantage news sentiment adapter."""

    base_url = "https://www.alphavantage.co/query"

    def __init__(self, *, api_key: str | None = None, timeout_seconds: int = 10) -> None:
        load_project_env()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.timeout_seconds = timeout_seconds

    def fetch_market_news(self) -> tuple[NewsArticle, ...]:
        """Fetch broad news."""

        return self.fetch_company_news("NIFTY")

    def fetch_company_news(self, symbol: str) -> tuple[NewsArticle, ...]:
        """Fetch symbol/topic news."""

        data = self._get({"function": "NEWS_SENTIMENT", "tickers": symbol})
        return self.normalize_response(data, symbol=symbol)

    def fetch_economic_calendar(self) -> tuple[NewsArticle, ...]:
        """Alpha Vantage economic calendar placeholder via broad news."""

        return self.fetch_market_news()

    def fetch_news_sentiment(self, symbol: str) -> dict[str, Any]:
        """Return raw sentiment response when available."""

        return self._get({"function": "NEWS_SENTIMENT", "tickers": symbol})

    def normalize_response(self, response: Any, *, symbol: str | None = None) -> tuple[NewsArticle, ...]:
        """Normalize Alpha Vantage response."""

        rows = response.get("feed", []) if isinstance(response, dict) else []
        articles = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            articles.append(
                article_from_mapping(
                    {
                        "headline": row.get("title"),
                        "summary": row.get("summary"),
                        "source": row.get("source"),
                        "url": row.get("url"),
                        "timestamp": row.get("time_published"),
                        "symbol": symbol,
                        "metadata": {"overall_sentiment_score": row.get("overall_sentiment_score")},
                    },
                    provider="alpha_vantage",
                )
            )
        return tuple(article for article in articles if article.headline)

    def _get(self, params: dict[str, Any]) -> Any:
        if not self.api_key:
            raise BrokerError("Alpha Vantage API key is missing")
        if requests is None:
            raise BrokerError("requests package is required for Alpha Vantage news")
        try:
            response = requests.get(self.base_url, params={**params, "apikey": self.api_key}, timeout=self.timeout_seconds)
            if response.status_code == 429:
                raise BrokerError("Alpha Vantage rate limit exceeded")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            LOGGER.warning("Alpha Vantage request failed", extra={"function": params.get("function")})
            raise BrokerError("Alpha Vantage request failed") from exc
