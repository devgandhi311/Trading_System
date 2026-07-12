"""Diagnose news integration across features, prediction, regime, and risk.

Run:
    python scripts/check_news_integration.py
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_pipeline import FeaturePipeline
from news import NewsScheduler
from risk.risk_manager import RiskManager
from utils.env_loader import get_secret_status, load_project_env


def main() -> int:
    """Print safe news integration status."""

    load_project_env(PROJECT_ROOT / ".env")
    scheduler = NewsScheduler()
    event_risk = scheduler.refresh_if_due(force=True)
    latest_sentiment = scheduler.latest_sentiment

    features = FeaturePipeline().generate_features(
        _sample_market_data(),
        dataset_name="news_integration_check",
        cache=False,
        latest_news_sentiment_score=latest_sentiment.sentiment_score if latest_sentiment else 0.0,
        latest_event_risk_score=event_risk.event_risk_score,
    ).features
    risk_source = inspect.getsource(RiskManager.validate)
    risk_considers_news = "event_risk_level" in risk_source and "news_event_risk" in risk_source

    print(f"FINNHUB key loaded: {get_secret_status('FINNHUB_API_KEY')}")
    print(f"news articles count: {len(scheduler.latest_articles)}")
    print(f"sentiment score: {latest_sentiment.sentiment_score if latest_sentiment else 0.0:.4f}")
    print(f"event risk level: {event_risk.event_risk_level}")
    print(f"feature has news_sentiment_score: {_yes_no('news_sentiment_score' in features.columns)}")
    print(f"feature has event_risk_score: {_yes_no('event_risk_score' in features.columns)}")
    print(f"risk manager considers news: {_yes_no(risk_considers_news)}")
    return 0


def _sample_market_data(rows: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = 20_000 + np.linspace(0, 150, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 8,
            "high": close + 30,
            "low": close - 30,
            "close": close,
            "volume": np.full(rows, 1_000_000),
        }
    )


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


if __name__ == "__main__":
    raise SystemExit(main())
