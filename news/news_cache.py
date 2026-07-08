"""Local JSON cache for news articles and risk results."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from news.news_config import NewsArticle, article_from_mapping
from utils.file_utils import ensure_parent_dir
from utils.time_utils import now_ist


class NewsCache:
    """Cache news data under data/news_cache."""

    def __init__(self, *, cache_dir: str | Path = "data/news_cache", expiry_minutes: int = 60) -> None:
        """Create cache."""

        self.cache_dir = Path(cache_dir)
        self.expiry = timedelta(minutes=expiry_minutes)

    def get_articles(self, key: str) -> tuple[NewsArticle, ...] | None:
        """Return cached articles if fresh."""

        path = self._path(key)
        if not path.exists() or now_ist() - datetime.fromtimestamp(path.stat().st_mtime, tz=now_ist().tzinfo) > self.expiry:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return tuple(article_from_mapping(item) for item in data.get("articles", []))

    def set_articles(self, key: str, articles: tuple[NewsArticle, ...] | list[NewsArticle]) -> Path:
        """Persist cached articles."""

        path = ensure_parent_dir(self._path(key))
        path.write_text(
            json.dumps({"cached_at": now_ist().isoformat(), "articles": [_serialize(article) for article in articles]}, indent=2),
            encoding="utf-8",
        )
        return path

    def _path(self, key: str) -> Path:
        safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in key)
        return self.cache_dir / f"{safe}.json"


def _serialize(value: Any) -> dict[str, Any]:
    data = asdict(value) if is_dataclass(value) and not isinstance(value, type) else dict(value)
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()
    return data
