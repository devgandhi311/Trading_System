"""News-driven event risk detector."""

from __future__ import annotations

from datetime import timedelta

from news.news_config import EventRiskResult, NewsArticle, NewsConfigLoader, SentimentResult, neutral_event_risk
from utils.time_utils import now_ist


class EventRiskDetector:
    """Convert news and sentiment into risk-filter decisions."""

    def __init__(self, *, config_loader: NewsConfigLoader | None = None) -> None:
        self.config = (config_loader or NewsConfigLoader()).load()

    def detect(
        self,
        cleaned_news: tuple[NewsArticle, ...] | list[NewsArticle],
        sentiment: SentimentResult | None,
        current_market_regime: object | None = None,
    ) -> EventRiskResult:
        """Detect event risk from cleaned news and sentiment."""

        if not self.config.enabled:
            return neutral_event_risk("News filter disabled")
        if not cleaned_news:
            return neutral_event_risk("No relevant news")
        risk_articles = self._risk_articles(tuple(cleaned_news))
        keyword_score = min(0.55, len(risk_articles) * 0.12)
        high_source_score = min(0.25, sum(1 for article in risk_articles if self._is_high_source(article.source)) * 0.08)
        sentiment_score = 0.0 if sentiment is None else max(0.0, -sentiment.sentiment_score) * 0.35
        recency_score = 0.2 if any(now_ist() - article.timestamp < timedelta(minutes=30) for article in risk_articles) else 0.0
        regime_score = 0.1 if "event" in str(current_market_regime or "").lower() else 0.0
        score = max(0.0, min(1.0, keyword_score + high_source_score + sentiment_score + recency_score + regime_score))
        level = self._level(score)
        rules = self.config.risk_rules
        trade_allowed = not (level in {"HIGH", "EXTREME"} and bool(rules.get("block_new_trades_on_high_event_risk", True)))
        multiplier = 0.5 if level == "MEDIUM" and bool(rules.get("reduce_position_size_on_medium_event_risk", True)) else 1.0
        if level in {"HIGH", "EXTREME"}:
            multiplier = 0.0
        minutes = int(rules.get("minimum_minutes_after_high_impact_news", 30)) if level in {"HIGH", "EXTREME"} else 0
        reason = f"News event risk {level}: {len(risk_articles)} relevant risk articles."
        if level == "EXTREME" and sentiment and sentiment.sentiment_score <= -0.6:
            reason += " Extreme negative news near market hours warning."
        return EventRiskResult(
            event_risk_score=score,
            event_risk_level=level,
            trade_allowed=trade_allowed,
            position_size_multiplier=multiplier,
            no_trade_minutes=minutes,
            reason=reason,
            top_risk_articles=tuple(risk_articles[:5]),
        )

    def _risk_articles(self, articles: tuple[NewsArticle, ...]) -> list[NewsArticle]:
        keywords = [keyword.lower() for keyword in self.config.event_risk_keywords]
        result = []
        for article in articles:
            text = f"{article.headline} {article.summary} {article.topic}".lower()
            if any(keyword.lower() in text for keyword in keywords):
                result.append(article)
        return result

    def _is_high_source(self, source: str) -> bool:
        return any(source.lower() == item.lower() for item in self.config.high_impact_sources)

    @staticmethod
    def _level(score: float) -> str:
        if score >= 0.85:
            return "EXTREME"
        if score >= 0.6:
            return "HIGH"
        if score >= 0.3:
            return "MEDIUM"
        return "LOW"
