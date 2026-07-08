"""Provider fallback news fetcher."""

from __future__ import annotations

import logging

from news.alpha_vantage_client import AlphaVantageClient
from news.finnhub_client import FinnhubClient
from news.marketaux_client import MarketauxClient
from news.news_cache import NewsCache
from news.news_config import NewsArticle, NewsConfigLoader
from utils.exceptions import BrokerError

LOGGER = logging.getLogger(__name__)


class NewsFetcher:
    """Fetch news using primary provider first, then backups."""

    def __init__(
        self,
        *,
        config_loader: NewsConfigLoader | None = None,
        providers: dict[str, object] | None = None,
        cache: NewsCache | None = None,
    ) -> None:
        self.config_loader = config_loader or NewsConfigLoader()
        self.config = self.config_loader.load()
        self.providers = providers or {
            "finnhub": FinnhubClient(),
            "alpha_vantage": AlphaVantageClient(),
            "marketaux": MarketauxClient(),
        }
        self.cache = cache or NewsCache(expiry_minutes=self.config.cache_expiry_minutes)
        self.last_error: str | None = None

    def fetch_all(self, *, use_cache: bool = True) -> tuple[NewsArticle, ...]:
        """Fetch market, symbol, and economic calendar news."""

        if not self.config.enabled:
            return ()
        cache_key = "all_news"
        if use_cache and self.config.cache_enabled:
            cached = self.cache.get_articles(cache_key)
            if cached is not None:
                return cached
        articles: list[NewsArticle] = []
        for provider_name in self._provider_order():
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            try:
                articles.extend(provider.fetch_market_news())
                for symbol in self.config.symbols:
                    articles.extend(provider.fetch_company_news(symbol))
                articles.extend(provider.fetch_economic_calendar())
                if articles:
                    result = tuple(articles)
                    if self.config.cache_enabled:
                        self.cache.set_articles(cache_key, result)
                    self.last_error = None
                    return result
                LOGGER.warning("News provider returned empty response", extra={"provider": provider_name})
            except BrokerError as exc:
                self.last_error = str(exc)
                LOGGER.warning("News provider failed", extra={"provider": provider_name, "error": type(exc).__name__})
        return ()

    def _provider_order(self) -> tuple[str, ...]:
        return (self.config.primary_provider, *self.config.backup_providers)
