"""Volatility scoring for regime detection."""

from __future__ import annotations

import pandas as pd


class VolatilityRegimeAnalyzer:
    """Score volatility intensity."""

    def score(self, features: pd.DataFrame) -> float:
        """Return volatility score in [0, 1]."""

        if features.empty:
            return 0.0
        percentile = features.get("volatility_percentile")
        if percentile is not None and not pd.isna(percentile.iloc[-1]):
            return float(max(0.0, min(1.0, percentile.iloc[-1])))
        realized = features.get("realized_volatility", pd.Series([0.0]))
        value = float(realized.iloc[-1] if not pd.isna(realized.iloc[-1]) else 0.0)
        return float(max(0.0, min(1.0, value)))
