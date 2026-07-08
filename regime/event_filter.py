"""Event-risk filters for no-trade regime detection."""

from __future__ import annotations

import pandas as pd

from news.news_config import EventRiskResult


class EventRiskFilter:
    """Score event-risk/no-trade context."""

    def score(self, features: pd.DataFrame, news_event_risk: EventRiskResult | None = None) -> float:
        """Return event-risk score in [0, 1]."""

        news_score = news_event_risk.event_risk_score if news_event_risk is not None else 0.0
        if features.empty:
            return news_score
        latest = features.iloc[-1]
        score = news_score
        if bool(latest.get("abnormal_candle_detection", False)):
            score += 0.35
        if bool(latest.get("range_expansion", False)):
            score += 0.25
        slippage = latest.get("slippage_estimate", 0.0)
        if pd.notna(slippage):
            score += min(float(slippage) * 5, 0.25)
        volume_spike = latest.get("volume_spike", 0.0)
        if pd.notna(volume_spike) and volume_spike > 2:
            score += 0.2
        return float(max(0.0, min(1.0, score)))
