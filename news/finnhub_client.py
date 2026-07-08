"""Finnhub news provider adapter."""

from __future__ import annotations

from datetime import date, timedelta
import logging
import os
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    requests = None  # type: ignore[assignment]

from news.news_config import NewsArticle, article_from_mapping
from utils.exceptions import BrokerError
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)


class FinnhubClient:
    """Finnhub client for market news, company news, calendar, and sentiment."""

    base_url = "https://finnhub.io/api/v1"

    def __init__(self, *, api_key: str | None = None, timeout_seconds: int = 10) -> None:
        """Create Finnhub client."""

        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self.timeout_seconds = timeout_seconds

    def fetch_market_news(self) -> tuple[NewsArticle, ...]:
        """Fetch general market news."""

        return self.normalize_response(self._get("/news", {"category": "general"}), endpoint="market_news")

    def fetch_company_news(self, symbol: str) -> tuple[NewsArticle, ...]:
        """Fetch company/symbol news."""

        today = date.today()
        params = {"symbol": symbol, "from": (today - timedelta(days=7)).isoformat(), "to": today.isoformat()}
        return self.normalize_response(self._get("/company-news", params), endpoint="company_news", symbol=symbol)

    def fetch_economic_calendar(self) -> tuple[NewsArticle, ...]:
        """Fetch economic-calendar-like entries when supported."""

        data = self._get("/calendar/economic", {})
        rows = data.get("economicCalendar", []) if isinstance(data, dict) else data
        return self.normalize_response(rows, endpoint="economic_calendar")

    def fetch_news_sentiment(self, symbol: str) -> dict[str, Any]:
        """Fetch provider sentiment for symbol."""

        response = self._get("/news-sentiment", {"symbol": symbol})
        return dict(response) if isinstance(response, dict) else {}

    def normalize_response(self, response: Any, *, endpoint: str = "", symbol: str | None = None) -> tuple[NewsArticle, ...]:
        """Normalize Finnhub response into NewsArticle items."""

        rows = response if isinstance(response, list) else response.get("data", []) if isinstance(response, dict) else []
        articles = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            articles.append(
                article_from_mapping(
                    {
                        "headline": row.get("headline") or row.get("event") or row.get("title"),
                        "summary": row.get("summary") or row.get("description"),
                        "source": row.get("source"),
                        "url": row.get("url"),
                        "timestamp": row.get("datetime") or row.get("time") or now_ist(),
                        "symbol": symbol,
                        "metadata": {"endpoint": endpoint},
                    },
                    provider="finnhub",
                )
            )
        return tuple(article for article in articles if article.headline)

    def _get(self, endpoint: str, params: dict[str, Any]) -> Any:
        if not self.api_key:
            raise BrokerError("Finnhub API key is missing")
        if requests is None:
            raise BrokerError("requests package is required for Finnhub news")
        safe_params = {**params, "token": self.api_key}
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=safe_params, timeout=self.timeout_seconds)
            if response.status_code == 429:
                raise BrokerError("Finnhub rate limit exceeded")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            LOGGER.warning("Finnhub request failed", extra={"endpoint": endpoint})
            raise BrokerError("Finnhub request failed") from exc
