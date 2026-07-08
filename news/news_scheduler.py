"""Simple interval scheduler for cached news refreshes."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

from news.news_cleaner import NewsCleaner
from news.news_config import EventRiskResult, NewsConfigLoader, NewsArticle, SentimentResult, neutral_event_risk
from news.news_fetcher import NewsFetcher
from news.news_sentiment import NewsSentiment
from news.event_risk_detector import EventRiskDetector
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)


class NewsScheduler:
    """Fetch news at configured intervals and cache latest analysis in memory."""

    def __init__(
        self,
        *,
        config_loader: NewsConfigLoader | None = None,
        fetcher: NewsFetcher | None = None,
        cleaner: NewsCleaner | None = None,
        sentiment: NewsSentiment | None = None,
        detector: EventRiskDetector | None = None,
    ) -> None:
        self.config_loader = config_loader or NewsConfigLoader()
        self.config = self.config_loader.load()
        self.fetcher = fetcher or NewsFetcher(config_loader=self.config_loader)
        self.cleaner = cleaner or NewsCleaner(config_loader=self.config_loader)
        self.sentiment = sentiment or NewsSentiment(config_loader=self.config_loader)
        self.detector = detector or EventRiskDetector(config_loader=self.config_loader)
        self.last_fetch_at: datetime | None = None
        self.latest_articles: tuple[NewsArticle, ...] = ()
        self.latest_sentiment: SentimentResult | None = None
        self.latest_event_risk: EventRiskResult = neutral_event_risk()

    def refresh_if_due(self, *, force: bool = False, current_market_regime: object | None = None) -> EventRiskResult:
        """Refresh news if interval elapsed."""

        if not self.config.enabled:
            self.latest_event_risk = neutral_event_risk("News disabled")
            return self.latest_event_risk
        due = self.last_fetch_at is None or now_ist() - self.last_fetch_at >= timedelta(minutes=self.config.fetch_interval_minutes)
        if not force and not due:
            return self.latest_event_risk
        articles = self.fetcher.fetch_all()
        if not articles and self.config.risk_rules.get("fail_safe_block_on_news_api_error", False):
            self.latest_event_risk = EventRiskResult(1.0, "EXTREME", False, 0.0, 30, "News API failed; fail-safe block enabled.", ())
            return self.latest_event_risk
        self.latest_articles = self.cleaner.clean(articles)
        self.latest_sentiment = self.sentiment.analyze(self.latest_articles)
        self.latest_event_risk = self.detector.detect(self.latest_articles, self.latest_sentiment, current_market_regime)
        self.last_fetch_at = now_ist()
        LOGGER.info("News refresh completed", extra={"articles": len(self.latest_articles), "risk": self.latest_event_risk.event_risk_level})
        return self.latest_event_risk
