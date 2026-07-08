"""News integration configuration and shared models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from utils.exceptions import ConfigError
from utils.time_utils import now_ist


@dataclass(frozen=True)
class NewsConfig:
    """Validated news configuration."""

    enabled: bool
    primary_provider: str
    backup_providers: tuple[str, ...]
    fetch_interval_minutes: int
    cache_enabled: bool
    cache_expiry_minutes: int
    symbols: tuple[str, ...]
    event_risk_keywords: tuple[str, ...]
    high_impact_sources: tuple[str, ...]
    sentiment_thresholds: dict[str, float]
    risk_rules: dict[str, Any]


@dataclass(frozen=True)
class NewsArticle:
    """Normalized news article."""

    headline: str
    source: str
    timestamp: datetime
    url: str = ""
    summary: str = ""
    symbol: str | None = None
    provider: str = ""
    topic: str = "macro"
    relevance_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SentimentResult:
    """News sentiment output."""

    sentiment_score: float
    sentiment_label: str
    confidence: float
    top_positive_news: tuple[NewsArticle, ...]
    top_negative_news: tuple[NewsArticle, ...]
    topic_wise_sentiment: dict[str, float]
    explanation: str


@dataclass(frozen=True)
class EventRiskResult:
    """Event risk output from news."""

    event_risk_score: float
    event_risk_level: str
    trade_allowed: bool
    position_size_multiplier: float
    no_trade_minutes: int
    reason: str
    top_risk_articles: tuple[NewsArticle, ...]


class NewsConfigLoader:
    """Load news_config.yaml without exposing secrets."""

    def __init__(self, path: str | Path = "config/news_config.yaml") -> None:
        """Create loader."""

        self.path = Path(path)
        self._config: NewsConfig | None = None

    def load(self, reload: bool = False) -> NewsConfig:
        """Load and validate news configuration."""

        if self._config is not None and not reload:
            return self._config
        if not self.path.exists():
            raise ConfigError(f"News config not found: {self.path}")
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError("news_config.yaml must contain a mapping")
        self._config = NewsConfig(
            enabled=bool(data.get("enabled", True)),
            primary_provider=_required_str(data, "primary_provider").lower(),
            backup_providers=tuple(str(item).lower() for item in _required_list(data, "backup_providers")),
            fetch_interval_minutes=_positive_int(data, "fetch_interval_minutes"),
            cache_enabled=bool(data.get("cache_enabled", True)),
            cache_expiry_minutes=_positive_int(data, "cache_expiry_minutes"),
            symbols=tuple(str(item).upper() for item in _required_list(data, "symbols")),
            event_risk_keywords=tuple(str(item) for item in _required_list(data, "event_risk_keywords")),
            high_impact_sources=tuple(str(item) for item in _required_list(data, "high_impact_sources")),
            sentiment_thresholds={str(key): float(value) for key, value in dict(data.get("sentiment_thresholds", {})).items()},
            risk_rules=dict(data.get("risk_rules", {})),
        )
        return self._config


def neutral_event_risk(reason: str = "Neutral news risk") -> EventRiskResult:
    """Return a neutral news risk result."""

    return EventRiskResult(
        event_risk_score=0.0,
        event_risk_level="LOW",
        trade_allowed=True,
        position_size_multiplier=1.0,
        no_trade_minutes=0,
        reason=reason,
        top_risk_articles=(),
    )


def article_from_mapping(data: dict[str, Any], *, provider: str = "") -> NewsArticle:
    """Create NewsArticle from a mapping."""

    timestamp = data.get("timestamp") or data.get("datetime") or data.get("published_at") or now_ist()
    if isinstance(timestamp, int | float):
        timestamp = datetime.fromtimestamp(timestamp, tz=now_ist().tzinfo)
    elif isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(now_ist().tzinfo)
        except ValueError:
            timestamp = now_ist()
    elif not isinstance(timestamp, datetime):
        timestamp = now_ist()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=now_ist().tzinfo)
    else:
        timestamp = timestamp.astimezone(now_ist().tzinfo)
    return NewsArticle(
        headline=str(data.get("headline") or data.get("title") or ""),
        source=str(data.get("source") or data.get("source_name") or ""),
        timestamp=timestamp,
        url=str(data.get("url") or ""),
        summary=str(data.get("summary") or data.get("description") or ""),
        symbol=str(data.get("symbol")) if data.get("symbol") else None,
        provider=provider or str(data.get("provider") or ""),
        topic=str(data.get("topic") or "macro"),
        relevance_score=float(data.get("relevance_score", 1.0) or 1.0),
        metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata"), dict) else {},
    )


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing news config string: {key}")
    return value.strip()


def _required_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"Missing news config list: {key}")
    return value


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = int(data.get(key, 0))
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value
