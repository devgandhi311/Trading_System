"""News report generation."""

from __future__ import annotations

from pathlib import Path

from news.news_config import EventRiskResult, NewsArticle, SentimentResult
from utils.file_utils import save_excel
from utils.time_utils import now_ist


class NewsReport:
    """Write news sentiment and event-risk reports."""

    def __init__(self, *, report_dir: str | Path = "reports/risk_reports") -> None:
        self.report_dir = Path(report_dir)

    def save(
        self,
        *,
        articles: tuple[NewsArticle, ...],
        sentiment: SentimentResult | None,
        event_risk: EventRiskResult,
        name: str | None = None,
    ) -> Path:
        """Save news report to Excel."""

        rows = []
        rows.append(
            {
                "timestamp": now_ist().isoformat(),
                "type": "summary",
                "headline": event_risk.reason,
                "source": "",
                "topic": "",
                "sentiment_score": sentiment.sentiment_score if sentiment else 0.0,
                "sentiment_label": sentiment.sentiment_label if sentiment else "NEUTRAL",
                "event_risk_score": event_risk.event_risk_score,
                "event_risk_level": event_risk.event_risk_level,
                "trade_allowed": event_risk.trade_allowed,
            }
        )
        for article in articles:
            rows.append(
                {
                    "timestamp": article.timestamp.isoformat(),
                    "type": "article",
                    "headline": article.headline,
                    "source": article.source,
                    "topic": article.topic,
                    "sentiment_score": "",
                    "sentiment_label": "",
                    "event_risk_score": "",
                    "event_risk_level": "",
                    "trade_allowed": "",
                }
            )
        stem = name or f"news_report_{now_ist().date().isoformat()}"
        return save_excel(rows, self.report_dir / f"{stem}.xlsx", sheet_name="News")
