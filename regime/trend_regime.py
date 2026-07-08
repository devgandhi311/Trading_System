"""Trend scoring for regime detection."""

from __future__ import annotations

import pandas as pd


class TrendRegimeAnalyzer:
    """Score trend direction from return and moving-average context."""

    def score(self, features: pd.DataFrame) -> float:
        """Return trend score in [-1, 1]."""

        if features.empty:
            return 0.0
        close = features["close"] if "close" in features else None
        rolling_mean = float(features.get("rolling_mean_return", pd.Series([0.0])).iloc[-1] or 0.0)
        recent_return = float(features.get("daily_return", pd.Series([0.0])).tail(5).mean() or 0.0)
        ma_score = 0.0
        if close is not None and len(close) >= 20:
            fast = close.tail(5).mean()
            slow = close.tail(20).mean()
            ma_score = (fast - slow) / slow if slow else 0.0
        raw = rolling_mean * 20 + recent_return * 10 + ma_score * 5
        return float(max(-1.0, min(1.0, raw)))
