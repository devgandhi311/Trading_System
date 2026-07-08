"""News cleaning, deduplication, and topic tagging."""

from __future__ import annotations

import re

from news.news_config import NewsArticle, NewsConfigLoader


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "RBI": ("rbi", "reserve bank", "policy"),
    "inflation": ("inflation", "cpi", "wpi"),
    "currency": ("rupee", "usdinr", "dollar", "currency"),
    "crude": ("crude", "oil", "brent", "wti"),
    "global risk": ("fed", "fomc", "war", "geopolitical", "global"),
    "earnings": ("earnings", "profit", "results"),
    "regulation": ("sebi", "tax", "budget", "regulation"),
    "exchange outage": ("nse outage", "market halt", "exchange outage"),
    "geopolitical": ("war", "geopolitical", "border"),
    "index-specific": ("nifty", "banknifty", "nse"),
    "macro": ("gdp", "growth", "macro", "economy"),
}


class NewsCleaner:
    """Clean raw news articles."""

    def __init__(self, *, config_loader: NewsConfigLoader | None = None) -> None:
        self.config = (config_loader or NewsConfigLoader()).load()

    def clean(self, articles: tuple[NewsArticle, ...] | list[NewsArticle]) -> tuple[NewsArticle, ...]:
        """Deduplicate, normalize, filter, and tag articles."""

        seen: set[str] = set()
        cleaned: list[NewsArticle] = []
        for article in articles:
            headline = _normalize_space(article.headline)
            if not headline:
                continue
            key = re.sub(r"[^a-z0-9]+", "", headline.lower())
            if key in seen:
                continue
            seen.add(key)
            source = self.clean_source(article.source)
            topic = self.tag_topic(headline, article.summary)
            if not self.is_relevant(headline, article.summary):
                continue
            cleaned.append(
                NewsArticle(
                    headline=headline,
                    source=source,
                    timestamp=article.timestamp,
                    url=article.url,
                    summary=_normalize_space(article.summary),
                    symbol=article.symbol,
                    provider=article.provider,
                    topic=topic,
                    relevance_score=article.relevance_score,
                    metadata=article.metadata,
                )
            )
        return tuple(sorted(cleaned, key=lambda item: item.timestamp, reverse=True))

    @staticmethod
    def clean_source(source: str) -> str:
        """Normalize source names."""

        return _normalize_space(source).replace(".com", "")

    def is_relevant(self, headline: str, summary: str = "") -> bool:
        """Return true when article contains configured symbols or risk keywords."""

        text = f"{headline} {summary}".lower()
        symbols = [symbol.lower() for symbol in self.config.symbols]
        keywords = [keyword.lower() for keyword in self.config.event_risk_keywords]
        return any(item.lower() in text for item in (*symbols, *keywords))

    @staticmethod
    def tag_topic(headline: str, summary: str = "") -> str:
        """Tag article topic."""

        text = f"{headline} {summary}".lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return topic
        return "macro"


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
