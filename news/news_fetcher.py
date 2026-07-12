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
        if not self._has_configured_provider_credentials():
            self.last_error = "News API keys are missing"
            LOGGER.warning("News API keys missing; using neutral news risk")
            return ()
        cache_key = "all_news"
        if use_cache and self.config.cache_enabled:
            cached = self.cache.get_articles(cache_key)
            if cached is not None and self._cache_matches_configured_providers(cached):
                self.last_error = None
                return cached
            if cached is not None:
                LOGGER.warning("Ignoring news cache with non-configured provider articles")
        provider_errors: list[str] = []
        for provider_name in self._provider_order():
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            provider_articles, errors = self._fetch_provider_articles(provider_name, provider)
            provider_errors.extend(errors)
            if provider_articles:
                result = tuple(provider_articles)
                if self.config.cache_enabled:
                    self.cache.set_articles(cache_key, result)
                self.last_error = None
                return result
            if errors:
                self.last_error = "; ".join(errors)
            else:
                self.last_error = f"{provider_name}: No articles returned"
                LOGGER.warning("News provider returned empty response", extra={"provider": provider_name})
        if provider_errors:
            self.last_error = "; ".join(provider_errors)
        return ()

    def _fetch_provider_articles(self, provider_name: str, provider: object) -> tuple[list[NewsArticle], list[str]]:
        """Fetch all supported provider endpoints without discarding partial success."""

        articles: list[NewsArticle] = []
        errors: list[str] = []
        calls = [("market_news", lambda: provider.fetch_market_news())]
        calls.extend((f"company_news:{symbol}", lambda symbol=symbol: provider.fetch_company_news(symbol)) for symbol in self.config.symbols)
        calls.append(("economic_calendar", lambda: provider.fetch_economic_calendar()))
        for endpoint_name, call in calls:
            try:
                articles.extend(call())
            except BrokerError as exc:
                message = f"{provider_name}.{endpoint_name}: {exc}"
                errors.append(message)
                LOGGER.warning(
                    "News provider endpoint failed",
                    extra={"provider": provider_name, "endpoint": endpoint_name, "error": str(exc)},
                )
            except AttributeError:
                continue
        return articles, errors

    def _provider_order(self) -> tuple[str, ...]:
        return (self.config.primary_provider, *self.config.backup_providers)

    def _cache_matches_configured_providers(self, articles: tuple[NewsArticle, ...]) -> bool:
        """Return true when cached articles came from configured real providers."""

        configured = set(self._provider_order())
        providers = {article.provider for article in articles if article.provider}
        return not providers or bool(providers.intersection(configured))

    def _has_configured_provider_credentials(self) -> bool:
        """Return whether at least one configured provider can authenticate.

        Test doubles and offline adapters may not expose an ``api_key`` attribute;
        those are treated as usable so unit tests can inject deterministic
        providers without environment secrets.
        """

        saw_api_key_provider = False
        for provider_name in self._provider_order():
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            if not hasattr(provider, "api_key"):
                return True
            saw_api_key_provider = True
            if getattr(provider, "api_key", None):
                return True
        return not saw_api_key_provider
