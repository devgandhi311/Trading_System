"""Top-level market regime detector."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from features.feature_pipeline import FeaturePipeline
from models import RegimeResult, RegimeType
from news.news_config import EventRiskResult
from regime.event_filter import EventRiskFilter
from regime.liquidity_regime import LiquidityRegimeAnalyzer
from regime.trend_regime import TrendRegimeAnalyzer
from regime.volatility_regime import VolatilityRegimeAnalyzer
from utils.time_utils import now_ist


class RegimeDetector:
    """Classify market regime without placing or implying orders."""

    def __init__(self, feature_pipeline: FeaturePipeline | None = None) -> None:
        """Create a regime detector."""

        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.trend = TrendRegimeAnalyzer()
        self.volatility = VolatilityRegimeAnalyzer()
        self.liquidity = LiquidityRegimeAnalyzer()
        self.event_filter = EventRiskFilter()

    def detect(
        self,
        data: pd.DataFrame,
        *,
        symbol: str = "NIFTY",
        news_event_risk: EventRiskResult | None = None,
    ) -> RegimeResult:
        """Detect one of the configured regimes from historical/features data."""

        if "smart_money_context_strength" not in data.columns:
            features = self.feature_pipeline.generate_features(data, cache=False).features
        else:
            features = data
        trend_score = self.trend.score(features)
        volatility_score = self.volatility.score(features)
        liquidity_score = self.liquidity.liquidity_score(features)
        trap_score = self.liquidity.trap_score(features)
        event_score = self.event_filter.score(features, news_event_risk=news_event_risk)

        regime_name = self._classify(trend_score, volatility_score, event_score, trap_score)
        allowed = regime_name not in {"Event-risk no-trade", "Liquidity-trap/fake-breakout"}
        if news_event_risk is not None and not news_event_risk.trade_allowed:
            allowed = False
        confidence = self._confidence(trend_score, volatility_score, liquidity_score, event_score, trap_score)
        preferred, forbidden = self._strategy_preferences(regime_name)
        explanation = (
            f"trend={trend_score:.2f}, volatility={volatility_score:.2f}, "
            f"liquidity={liquidity_score:.2f}, event_risk={event_score:.2f}, trap={trap_score:.2f}"
        )
        return RegimeResult(
            symbol=symbol,
            timestamp=now_ist(),
            regime=self._regime_type(regime_name),
            confidence=confidence,
            lookback_bars=len(features),
            features={
                "trend_score": trend_score,
                "volatility_score": volatility_score,
                "liquidity_score": liquidity_score,
                "event_risk_score": event_score,
                "trap_score": trap_score,
            },
            regime_name=regime_name,
            trend_score=trend_score,
            volatility_score=volatility_score,
            liquidity_score=liquidity_score,
            event_risk_score=event_score,
            confidence_score=confidence,
            allowed_to_trade=allowed,
            preferred_strategy_types=preferred,
            forbidden_strategy_types=forbidden,
            explanation=explanation,
        )

    @staticmethod
    def _classify(trend_score: float, volatility_score: float, event_score: float, trap_score: float) -> str:
        if event_score >= 0.6:
            return "Event-risk no-trade"
        if trap_score >= 0.6 and abs(trend_score) < 0.25:
            return "Liquidity-trap/fake-breakout"
        vol = "high volatility" if volatility_score >= 0.5 else "low volatility"
        if trend_score > 0.15:
            return f"Bullish {vol}"
        if trend_score < -0.15:
            return f"Bearish {vol}"
        return f"Sideways {vol}"

    @staticmethod
    def _confidence(
        trend_score: float,
        volatility_score: float,
        liquidity_score: float,
        event_score: float,
        trap_score: float,
    ) -> float:
        base = 0.35 + abs(trend_score) * 0.25 + abs(volatility_score - 0.5) * 0.2 + liquidity_score * 0.2
        penalty = event_score * 0.25 + trap_score * 0.25
        return float(max(0.0, min(1.0, base - penalty)))

    @staticmethod
    def _strategy_preferences(regime_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if regime_name in {"Event-risk no-trade", "Liquidity-trap/fake-breakout"}:
            return (), ("all_live_entries",)
        if regime_name.startswith("Bullish"):
            return ("momentum_breakout", "bull_put_spread"), ("bearish_directional",)
        if regime_name.startswith("Bearish"):
            return ("mean_reversion", "bear_call_spread"), ("bullish_directional",)
        if "high volatility" in regime_name:
            return ("iron_condor", "volatility_scalping"), ("naked_long_premium",)
        return ("range_mean_reversion", "iron_condor"), ("breakout_only",)

    @staticmethod
    def _regime_type(regime_name: str) -> RegimeType:
        if regime_name.startswith("Bullish") or regime_name.startswith("Bearish"):
            return RegimeType.TRENDING
        if "high volatility" in regime_name:
            return RegimeType.HIGH_VOLATILITY
        if "low volatility" in regime_name:
            return RegimeType.LOW_VOLATILITY
        return RegimeType.RANGE_BOUND
