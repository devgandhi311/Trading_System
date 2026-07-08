"""Local news sentiment scoring."""

from __future__ import annotations

from datetime import timedelta
import math

from news.news_config import NewsArticle, NewsConfigLoader, SentimentResult
from utils.time_utils import now_ist

POSITIVE_WORDS = {"gain", "rally", "growth", "cooling", "cut", "boost", "record", "surge", "strong", "relief", "stable"}
NEGATIVE_WORDS = {"fall", "crash", "war", "inflation", "hike", "crisis", "outage", "halt", "weak", "risk", "plunge", "default"}


class NewsSentiment:
    """Calculate headline, source, recency, and topic sentiment."""

    def __init__(self, *, config_loader: NewsConfigLoader | None = None) -> None:
        self.config = (config_loader or NewsConfigLoader()).load()
        self._vader = _load_vader()

    def analyze(self, articles: tuple[NewsArticle, ...] | list[NewsArticle]) -> SentimentResult:
        """Analyze news sentiment."""

        if not articles:
            return SentimentResult(0.0, "NEUTRAL", 0.0, (), (), {}, "No news articles available.")
        scored = [(article, self._article_score(article)) for article in articles]
        weighted = []
        for article, score in scored:
            weighted.append(score * self._source_weight(article.source) * self._recency_weight(article))
        sentiment_score = max(-1.0, min(1.0, sum(weighted) / max(1, len(weighted))))
        topic_scores: dict[str, list[float]] = {}
        for article, score in scored:
            topic_scores.setdefault(article.topic, []).append(score)
        topic_wise = {topic: sum(values) / len(values) for topic, values in topic_scores.items()}
        positives = tuple(article for article, score in sorted(scored, key=lambda item: item[1], reverse=True)[:3] if score > 0)
        negatives = tuple(article for article, score in sorted(scored, key=lambda item: item[1])[:3] if score < 0)
        confidence = min(1.0, 0.3 + len(articles) / 20.0 + abs(sentiment_score) * 0.4)
        return SentimentResult(
            sentiment_score=sentiment_score,
            sentiment_label=self.label(sentiment_score),
            confidence=confidence,
            top_positive_news=positives,
            top_negative_news=negatives,
            topic_wise_sentiment=topic_wise,
            explanation=f"{len(articles)} articles analyzed; score={sentiment_score:.2f}.",
        )

    def label(self, score: float) -> str:
        """Return sentiment label."""

        thresholds = self.config.sentiment_thresholds
        if score <= thresholds.get("very_negative", -0.6):
            return "VERY_NEGATIVE"
        if score <= thresholds.get("negative", -0.25):
            return "NEGATIVE"
        if score >= thresholds.get("very_positive", 0.6):
            return "VERY_POSITIVE"
        if score >= thresholds.get("positive", 0.25):
            return "POSITIVE"
        return "NEUTRAL"

    def _article_score(self, article: NewsArticle) -> float:
        text = f"{article.headline} {article.summary}".lower()
        if self._vader is not None:
            return float(self._vader.polarity_scores(text)["compound"])
        try:
            from textblob import TextBlob

            return float(TextBlob(text).sentiment.polarity)
        except Exception:
            positives = sum(1 for word in POSITIVE_WORDS if word in text)
            negatives = sum(1 for word in NEGATIVE_WORDS if word in text)
            if positives == negatives:
                return 0.0
            return max(-1.0, min(1.0, (positives - negatives) / max(1, positives + negatives)))

    def _source_weight(self, source: str) -> float:
        return 1.25 if any(source.lower() == item.lower() for item in self.config.high_impact_sources) else 1.0

    @staticmethod
    def _recency_weight(article: NewsArticle) -> float:
        age = now_ist() - article.timestamp
        if age < timedelta(minutes=30):
            return 1.25
        hours = max(age.total_seconds() / 3600.0, 0.0)
        return max(0.3, math.exp(-hours / 24.0))


def _load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except Exception:
        return None
