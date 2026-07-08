"""Liquidity and fake-breakout scoring for regime detection."""

from __future__ import annotations

import pandas as pd


class LiquidityRegimeAnalyzer:
    """Score liquidity health and trap risk."""

    def liquidity_score(self, features: pd.DataFrame) -> float:
        """Return liquidity score in [0, 1]."""

        if features.empty or "liquidity_score" not in features:
            return 0.5
        value = features["liquidity_score"].iloc[-1]
        return 0.5 if pd.isna(value) else float(max(0.0, min(1.0, value)))

    def trap_score(self, features: pd.DataFrame) -> float:
        """Return liquidity-trap/fake-breakout score in [0, 1]."""

        if features.empty:
            return 0.0
        latest = features.iloc[-1]
        failed_breakout = bool(latest.get("failed_breakout_detected", False))
        sweep = any(
            bool(latest.get(column, False))
            for column in (
                "previous_day_high_sweep",
                "previous_day_low_sweep",
                "liquidity_sweep_above_previous_high_detected",
                "liquidity_sweep_below_previous_low_detected",
            )
        )
        liquidity = self.liquidity_score(features)
        failed_strength = float(latest.get("failed_breakout_strength", 0.0) or 0.0)
        sweep_strength = max(
            float(latest.get("liquidity_sweep_above_previous_high_strength", 0.0) or 0.0),
            float(latest.get("liquidity_sweep_below_previous_low_strength", 0.0) or 0.0),
        )
        score = 0.0
        if failed_breakout:
            score += 0.45 + failed_strength * 0.4
        if sweep and liquidity < 0.4:
            score += 0.25 + sweep_strength * 0.25
        return float(max(0.0, min(1.0, score)))
