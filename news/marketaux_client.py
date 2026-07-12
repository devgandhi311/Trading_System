"""Marketaux news backup adapter."""

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


class MarketauxClient:
    """Marketaux news adapter."""

    base_url = "https://api.marketaux.com/v1/news/all"

    def __init__(self, *, api_key: str | None = None, timeout_seconds: int = 10) -> None:
        load_project_env()
        self.api_key = api_key or os.getenv("MARKETAUX_API_KEY")
        self.timeout_seconds = timeout_seconds

    def fetch_market_news(self) -> tuple[NewsArticle, ...]:
        """Fetch broad India market news."""

        return self.fetch_company_news("NIFTY")

    def fetch_company_news(self, symbol: str) -> tuple[NewsArticle, ...]:
        """Fetch symbol news."""

        data = self._get({"symbols": symbol, "language": "en"})
        return self.normalize_response(data, symbol=symbol)

    def fetch_economic_calendar(self) -> tuple[NewsArticle, ...]:
        """Marketaux economic calendar placeholder via broad news."""

        return self.fetch_market_news()

    def fetch_news_sentiment(self, symbol: str) -> dict[str, Any]:
        """Return raw sentiment-like response."""

        return self._get({"symbols": symbol, "language": "en"})

    def normalize_response(self, response: Any, *, symbol: str | None = None) -> tuple[NewsArticle, ...]:
        """Normalize Marketaux response."""

        rows = response.get("data", []) if isinstance(response, dict) else []
        articles = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            articles.append(
                article_from_mapping(
                    {
                        "headline": row.get("title"),
                        "summary": row.get("description"),
                        "source": (row.get("source") or {}).get("name") if isinstance(row.get("source"), dict) else row.get("source"),
                        "url": row.get("url"),
                        "timestamp": row.get("published_at"),
                        "symbol": symbol,
                        "metadata": {"entities": row.get("entities", [])},
                    },
                    provider="marketaux",
                )
            )
        return tuple(article for article in articles if article.headline)

    def _get(self, params: dict[str, Any]) -> Any:
        if not self.api_key:
            raise BrokerError("Marketaux API key is missing")
        if requests is None:
            raise BrokerError("requests package is required for Marketaux news")
        try:
            response = requests.get(self.base_url, params={**params, "api_token": self.api_key}, timeout=self.timeout_seconds)
            if response.status_code == 429:
                raise BrokerError("Marketaux rate limit exceeded")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            LOGGER.warning("Marketaux request failed")
            raise BrokerError("Marketaux request failed") from exc
