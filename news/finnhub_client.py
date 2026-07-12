"""Finnhub news provider adapter."""

from __future__ import annotations

from datetime import date, timedelta
import logging
import os
from typing import Any
from urllib.parse import urlencode

try:
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    requests = None  # type: ignore[assignment]

from news.news_config import NewsArticle, article_from_mapping
from utils.exceptions import BrokerError
from utils.env_loader import load_project_env
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)


class FinnhubRequestError(BrokerError):
    """Finnhub failure that may carry a safe HTTP status code."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FinnhubClient:
    """Finnhub client for market news, company news, calendar, and sentiment."""

    base_url = "https://finnhub.io/api/v1"

    def __init__(self, *, api_key: str | None = None, timeout_seconds: int = 10) -> None:
        """Create Finnhub client."""

        self.api_key = api_key or self._load_api_key()
        self.timeout_seconds = timeout_seconds

    def fetch_market_news(self) -> tuple[NewsArticle, ...]:
        """Fetch general market news."""

        _status_code, articles = self.fetch_general_market_news()
        return articles

    def fetch_general_market_news(self) -> tuple[int, tuple[NewsArticle, ...]]:
        """Fetch Finnhub general market news with safe HTTP status metadata."""

        response = self._request("/news", {"category": "general"})
        data = self._decode_json(response, endpoint="/news")
        articles = self.normalize_response(data, endpoint="market_news")
        if not articles:
            LOGGER.warning("Finnhub returned empty response", extra={"endpoint": "/news", "status": response.status_code})
            raise FinnhubRequestError("Finnhub empty response: no articles returned", status_code=response.status_code)
        return response.status_code, articles

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
        response = self._request(endpoint, params)
        return self._decode_json(response, endpoint=endpoint)

    def _request(self, endpoint: str, params: dict[str, Any]) -> Any:
        if not self.api_key:
            LOGGER.warning("Finnhub API key missing")
            raise FinnhubRequestError("Finnhub API key is missing")
        if requests is None:
            raise FinnhubRequestError("requests package is required for Finnhub news")
        safe_params = {**params, "token": self.api_key}
        headers = {"Accept": "application/json", "User-Agent": "nifty50-trading-news/1.0"}
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                params=safe_params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            LOGGER.info("Finnhub HTTP response", extra={"endpoint": endpoint, "status": response.status_code})
            if response.status_code in {401, 403, 429}:
                raise FinnhubRequestError(
                    _finnhub_status_reason(response.status_code, response.text),
                    status_code=response.status_code,
                )
            response.raise_for_status()
            return response
        except requests.Timeout as exc:
            LOGGER.warning("Finnhub request timed out", extra={"endpoint": endpoint, "timeout": self.timeout_seconds})
            raise FinnhubRequestError("Finnhub network timeout") from exc
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            details = _redact_secret(str(exc), self.api_key)
            reason = f"Finnhub request failed: HTTP {status}" if status else f"Finnhub request failed: {details}"
            LOGGER.warning("Finnhub request failed", extra={"endpoint": endpoint, "status": status})
            raise FinnhubRequestError(reason, status_code=status) from exc

    def _decode_json(self, response: Any, *, endpoint: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            LOGGER.warning("Finnhub invalid response", extra={"endpoint": endpoint, "status": response.status_code})
            raise FinnhubRequestError("Finnhub invalid response: expected JSON", status_code=response.status_code) from exc

    def sanitized_url(self, endpoint: str, params: dict[str, Any]) -> str:
        """Return request URL with token redacted for diagnostics."""

        query = urlencode({**params, "token": "<redacted>"})
        return f"{self.base_url}{endpoint}?{query}"

    @staticmethod
    def _load_api_key() -> str | None:
        """Load Finnhub key through the project secret system."""

        load_project_env()
        try:
            from utils.config_loader import ConfigLoader

            return ConfigLoader().get_secret("FINNHUB_API_KEY", required=False)
        except Exception as exc:
            LOGGER.warning("Finnhub key lookup via ConfigLoader failed", extra={"error": type(exc).__name__})
            return os.getenv("FINNHUB_API_KEY")


def _finnhub_status_reason(status_code: int, response_text: str) -> str:
    body = (response_text or "").strip()[:200]
    if status_code == 401:
        return f"Finnhub 401 Unauthorized: {body}"
    if status_code == 403:
        return f"Finnhub 403 Invalid API key or forbidden: {body}"
    if status_code == 429:
        return f"Finnhub 429 Rate limit exceeded: {body}"
    return f"Finnhub HTTP {status_code}: {body}"


def _redact_secret(value: str, secret: str | None) -> str:
    """Redact an API secret from diagnostic text."""

    if not secret:
        return value
    return value.replace(secret, "<redacted>")
