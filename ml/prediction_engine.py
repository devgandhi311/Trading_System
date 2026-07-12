"""Prediction engine that combines features, regime, and option-chain context."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Mapping

from ml.explainability import ExplainabilityEngine
from ml.model_registry import ModelRegistry
from models import OptionChainSnapshot, PredictionResult, RegimeResult, RegimeType, StrategyType
from utils.config_loader import ConfigLoader
from utils.math_utils import clamp
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)


class PredictionEngine:
    """Create advisory predictions without placing trades."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        model_registry: ModelRegistry | None = None,
        explainability_engine: ExplainabilityEngine | None = None,
        models: Mapping[str, Any] | None = None,
    ) -> None:
        """Create an engine with optional preloaded models."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.model_registry = model_registry or ModelRegistry(config_loader=self.config_loader)
        self.explainability = explainability_engine or ExplainabilityEngine()
        self.models = dict(models or {})
        self._last_prediction_sources: dict[str, str] = {}
        self._prediction_count = 0
        self._fallback_prediction_count = 0
        self._model_prediction_count = 0

    def predict(
        self,
        latest_features: Mapping[str, Any],
        current_regime: RegimeResult | RegimeType | str,
        option_chain_snapshot: OptionChainSnapshot | None = None,
        *,
        underlying: str | None = None,
        timestamp: datetime | None = None,
        news_sentiment: Any | None = None,
        news_event_risk: Any | None = None,
    ) -> PredictionResult:
        """Return a PredictionResult from latest features, regime, and option-chain context."""

        regime = self._normalize_regime(current_regime)
        regime_name = self._regime_name(current_regime, regime)
        underlying_symbol = underlying or self._underlying(latest_features, current_regime, option_chain_snapshot)
        spot = self._spot_price(latest_features, option_chain_snapshot)

        self._last_prediction_sources = {}
        bullish_probability, bearish_probability, sideways_probability = self._direction_probabilities(latest_features)
        expected_move_percent = self._expected_move_percent(latest_features, option_chain_snapshot)
        expected_range_low = spot * (1.0 - expected_move_percent / 100.0) if spot else None
        expected_range_high = spot * (1.0 + expected_move_percent / 100.0) if spot else None
        volatility_state = self._volatility_state(latest_features)
        breakout_failure_probability = self._breakout_failure_probability(latest_features)
        confidence = self._confidence(
            bullish_probability,
            bearish_probability,
            sideways_probability,
            latest_features,
            current_regime,
        )
        news_score = _as_float(getattr(news_sentiment, "sentiment_score", None))
        if news_score is None:
            news_score = _as_float(latest_features.get("news_sentiment_score"))
        if news_score is not None:
            confidence = clamp(confidence + news_score * 0.05, 0.0, 1.0)
        if getattr(news_event_risk, "event_risk_level", "").upper() in {"HIGH", "EXTREME"}:
            confidence = clamp(confidence - 0.2, 0.0, 1.0)
        expected_return_percent = self._expected_return_percent(
            bullish_probability,
            bearish_probability,
            expected_move_percent,
        )
        recommended_strategy = self._recommended_strategy(
            latest_features=latest_features,
            regime_name=regime_name,
            bullish_probability=bullish_probability,
            bearish_probability=bearish_probability,
            sideways_probability=sideways_probability,
            expected_move_percent=expected_move_percent,
            volatility_state=volatility_state,
            confidence=confidence,
        )
        trade_allowed = self._trade_allowed(current_regime, confidence, recommended_strategy)
        reason = self._reason(
            trade_allowed=trade_allowed,
            confidence=confidence,
            regime_name=regime_name,
            recommended_strategy=recommended_strategy,
        )
        predicted_direction = self._predicted_direction(
            bullish_probability,
            bearish_probability,
            sideways_probability,
        )
        iv_percentile = _as_float(latest_features.get("iv_percentile"))
        explanation = self.explainability.build(
            latest_features,
            confidence=confidence,
            bullish_probability=bullish_probability,
            bearish_probability=bearish_probability,
            sideways_probability=sideways_probability,
            breakout_failure_probability=breakout_failure_probability,
            trade_allowed=trade_allowed,
            strategy_type=recommended_strategy.value if recommended_strategy else None,
            regime_name=regime_name,
            expected_move_percent=expected_move_percent,
            iv_percentile=iv_percentile,
        )

        prediction_source = self._prediction_source(latest_features)
        model_name, model_version = self._model_metadata(prediction_source)
        features_used_count = self._features_used_count(latest_features)
        expected_expiry_distribution = _expected_expiry_distribution(
            spot,
            expected_return_percent,
            expected_move_percent,
        )
        expected_move_points = spot * expected_move_percent / 100.0 if spot else None
        expected_expiry_price = (
            expected_expiry_distribution.get("mean")
            if expected_expiry_distribution is not None
            else None
        )
        distribution_confidence = _as_float(latest_features.get("distribution_confidence"))
        if distribution_confidence is None and expected_expiry_distribution is not None:
            distribution_confidence = confidence
        self._record_prediction(prediction_source)
        LOGGER.debug(
            "Prediction generated",
            extra={
                "underlying": underlying_symbol,
                "regime": regime_name,
                "confidence": confidence,
                "trade_allowed": trade_allowed,
                "prediction_source": prediction_source,
            },
        )
        return PredictionResult(
            symbol=underlying_symbol,
            underlying=underlying_symbol,
            timestamp=timestamp or now_ist(),
            predicted_direction=predicted_direction,
            confidence=confidence,
            confidence_score=confidence,
            horizon_minutes=375,
            regime=regime,
            expected_return_percent=expected_return_percent,
            expected_volatility_percent=expected_move_percent,
            prediction_source=prediction_source,
            model_name=model_name,
            model_version=model_version,
            features_used_count=features_used_count,
            bullish_probability=bullish_probability,
            bearish_probability=bearish_probability,
            sideways_probability=sideways_probability,
            expected_move_percent=expected_move_percent,
            expected_move_points=expected_move_points,
            expected_expiry_price=expected_expiry_price,
            distribution_confidence=distribution_confidence,
            expected_range_low=expected_range_low,
            expected_range_high=expected_range_high,
            volatility_state=volatility_state,
            breakout_failure_probability=breakout_failure_probability,
            recommended_strategy_type=recommended_strategy.value if recommended_strategy else None,
            trade_allowed=trade_allowed,
            reason=reason,
            metadata={
                "explainability": explanation.as_dict(),
                "option_chain_expiry": str(option_chain_snapshot.expiry) if option_chain_snapshot else None,
                "source": prediction_source,
                "prediction_source": prediction_source,
                "model_name": model_name,
                "model_version": model_version,
                "features_used_count": features_used_count,
                "expected_expiry_distribution": expected_expiry_distribution,
                "expected_move_points": expected_move_points,
                "expected_expiry_price": expected_expiry_price,
                "distribution_confidence": distribution_confidence,
                "prediction_sources": dict(self._last_prediction_sources),
                "news_sentiment_score": news_score,
                "event_risk_level": getattr(news_event_risk, "event_risk_level", None),
                "event_risk_score": getattr(news_event_risk, "event_risk_score", None),
            },
        )

    def _direction_probabilities(self, features: Mapping[str, Any]) -> tuple[float, float, float]:
        model = self.models.get("direction")
        if model is not None:
            probabilities = _predict_probabilities(model, features)
            if probabilities:
                self._last_prediction_sources["direction"] = "trained model"
                return _normalize_three(probabilities)
            self._last_prediction_sources["direction"] = "fallback model"
        else:
            self._last_prediction_sources["direction"] = "fallback model"

        return_value = _first_float(features, "daily_return", "log_return", "intraday_return", default=0.0)
        trend_score = _first_float(features, "trend_score", "rolling_mean_return", default=return_value)
        volatility = abs(_first_float(features, "rolling_volatility", "realized_volatility", default=0.01))
        directional_edge = clamp((trend_score * 35.0) + (return_value * 20.0), -0.35, 0.35)
        sideways = clamp(0.34 + max(0.0, 0.04 - volatility) * 4.0, 0.15, 0.65)
        remaining = 1.0 - sideways
        bullish = remaining * (0.5 + directional_edge)
        bearish = remaining - bullish
        return _normalize_three((bullish, bearish, sideways))

    def _expected_move_percent(
        self,
        features: Mapping[str, Any],
        option_chain_snapshot: OptionChainSnapshot | None,
    ) -> float:
        model = self.models.get("volatility")
        if model is not None:
            prediction = _predict_scalar(model, features)
            if prediction is not None:
                self._last_prediction_sources["volatility"] = "trained model"
                return clamp(abs(prediction), 0.05, 20.0)
            self._last_prediction_sources["volatility"] = "fallback model"
        else:
            self._last_prediction_sources["volatility"] = "fallback model"
        explicit = _first_float(
            features,
            "expected_move_from_iv",
            "expected_move_from_realized_volatility",
            "expected_range",
            default=0.0,
        )
        if explicit > 0:
            return clamp(explicit, 0.05, 20.0)
        iv_values = [
            contract.implied_volatility
            for contract in (option_chain_snapshot.contracts if option_chain_snapshot else ())
            if contract.implied_volatility is not None
        ]
        if iv_values:
            return clamp(sum(iv_values) / len(iv_values) / 16.0, 0.05, 20.0)
        realized = _first_float(features, "realized_volatility", "rolling_volatility", default=0.01)
        return clamp(abs(realized) * 100.0, 0.05, 20.0)

    def _breakout_failure_probability(self, features: Mapping[str, Any]) -> float:
        model = self.models.get("signal")
        if model is not None:
            probabilities = _predict_probabilities(model, features)
            if probabilities:
                self._last_prediction_sources["signal"] = "trained model"
                return clamp(probabilities[-1], 0.0, 1.0)
            self._last_prediction_sources["signal"] = "fallback model"
        else:
            self._last_prediction_sources["signal"] = "fallback model"
        failed_breakout = _first_float(features, "failed_breakout_detected", "failed_breakout_strength", default=0.0)
        range_compression = _first_float(features, "range_compression", default=0.0)
        liquidity_sweep = max(
            _first_float(features, "liquidity_sweep_above_previous_high_strength", default=0.0),
            _first_float(features, "liquidity_sweep_below_previous_low_strength", default=0.0),
        )
        return clamp(0.15 + failed_breakout * 0.55 + range_compression * 0.15 + liquidity_sweep * 0.15, 0.0, 1.0)

    def _volatility_state(self, features: Mapping[str, Any]) -> str:
        percentile = _first_float(features, "volatility_percentile", "iv_percentile", default=50.0)
        if percentile >= 70.0:
            return "expanding"
        if percentile <= 30.0:
            return "contracting"
        return "normal"

    def _confidence(
        self,
        bullish_probability: float,
        bearish_probability: float,
        sideways_probability: float,
        features: Mapping[str, Any],
        current_regime: RegimeResult | RegimeType | str,
    ) -> float:
        probabilities = sorted((bullish_probability, bearish_probability, sideways_probability), reverse=True)
        separation = probabilities[0] - probabilities[1]
        regime_confidence = _as_float(getattr(current_regime, "confidence_score", None))
        if regime_confidence is None:
            regime_confidence = _as_float(getattr(current_regime, "confidence", None))
        liquidity = _first_float(features, "liquidity_score", default=0.6)
        return clamp(0.35 + separation * 0.8 + (regime_confidence or 0.5) * 0.2 + liquidity * 0.1, 0.0, 1.0)

    def _recommended_strategy(
        self,
        *,
        latest_features: Mapping[str, Any],
        regime_name: str,
        bullish_probability: float,
        bearish_probability: float,
        sideways_probability: float,
        expected_move_percent: float,
        volatility_state: str,
        confidence: float,
    ) -> StrategyType | None:
        if confidence < self.config.strategy.min_confidence_score:
            return None
        allowed = {item for item in self.config.strategy.allowed_strategies}
        event_risk_level = str(latest_features.get("event_risk_level") or "").upper()
        iv_percentile = _first_float(latest_features, "iv_percentile", "volatility_percentile", default=50.0)
        expected_move_credit_allowed = "expected_move_credit_spread" in allowed
        if sideways_probability >= max(bullish_probability, bearish_probability) and "mean_reversion" in allowed:
            if (
                expected_move_credit_allowed
                and event_risk_level not in {"HIGH", "EXTREME"}
                and iv_percentile >= 55.0
                and expected_move_percent > 0
            ):
                return StrategyType.EXPECTED_MOVE_CREDIT_SPREAD
            return StrategyType.MEAN_REVERSION
        if (
            expected_move_credit_allowed
            and event_risk_level not in {"HIGH", "EXTREME"}
            and iv_percentile >= 55.0
            and expected_move_percent >= 0.5
        ):
            return StrategyType.EXPECTED_MOVE_CREDIT_SPREAD
        if volatility_state == "expanding" and "volatility_scalping" in allowed:
            return StrategyType.VOLATILITY_SCALPING
        if expected_move_percent >= 0.6 and "momentum_breakout" in allowed:
            return StrategyType.MOMENTUM_BREAKOUT
        if "bullish" in regime_name.lower() or "bearish" in regime_name.lower():
            if "momentum_breakout" in allowed:
                return StrategyType.MOMENTUM_BREAKOUT
        if "mean_reversion" in allowed:
            return StrategyType.MEAN_REVERSION
        return None

    def _trade_allowed(
        self,
        current_regime: RegimeResult | RegimeType | str,
        confidence: float,
        recommended_strategy: StrategyType | None,
    ) -> bool:
        regime_allowed = bool(getattr(current_regime, "allowed_to_trade", True))
        return bool(
            regime_allowed
            and recommended_strategy is not None
            and confidence >= self.config.strategy.min_confidence_score
        )

    def _reason(
        self,
        *,
        trade_allowed: bool,
        confidence: float,
        regime_name: str,
        recommended_strategy: StrategyType | None,
    ) -> str:
        if not trade_allowed:
            if recommended_strategy is None:
                return "No strategy passed confidence, regime, and configuration filters."
            return f"Trade blocked by filters despite {recommended_strategy.value} suggestion."
        return f"{recommended_strategy.value} allowed for {regime_name} at {confidence:.1%} confidence."

    @staticmethod
    def _normalize_regime(regime: RegimeResult | RegimeType | str) -> RegimeType:
        if isinstance(regime, RegimeResult):
            return regime.regime
        if isinstance(regime, RegimeType):
            return regime
        normalized = regime.lower().replace(" ", "_").replace("-", "_")
        for regime_type in RegimeType:
            if normalized == regime_type.value:
                return regime_type
        if "trend" in normalized or "bullish" in normalized or "bearish" in normalized:
            return RegimeType.TRENDING
        if "sideways" in normalized or "range" in normalized:
            return RegimeType.RANGE_BOUND
        if "high_vol" in normalized:
            return RegimeType.HIGH_VOLATILITY
        if "low_vol" in normalized:
            return RegimeType.LOW_VOLATILITY
        return RegimeType.UNKNOWN

    @staticmethod
    def _regime_name(regime: RegimeResult | RegimeType | str, normalized: RegimeType) -> str:
        if isinstance(regime, RegimeResult):
            return regime.regime_name or regime.regime.value
        if isinstance(regime, RegimeType):
            return regime.value
        return str(regime) or normalized.value

    @staticmethod
    def _underlying(
        features: Mapping[str, Any],
        regime: RegimeResult | RegimeType | str,
        option_chain_snapshot: OptionChainSnapshot | None,
    ) -> str:
        if option_chain_snapshot:
            return option_chain_snapshot.underlying
        symbol = features.get("symbol") or features.get("underlying") or getattr(regime, "symbol", None)
        return str(symbol or "NIFTY")

    @staticmethod
    def _spot_price(features: Mapping[str, Any], option_chain_snapshot: OptionChainSnapshot | None) -> float | None:
        if option_chain_snapshot:
            return option_chain_snapshot.spot_price
        return _first_float(features, "close", "spot_price", "last_price", default=None)

    @staticmethod
    def _predicted_direction(bullish: float, bearish: float, sideways: float) -> str:
        return max((("bullish", bullish), ("bearish", bearish), ("sideways", sideways)), key=lambda item: item[1])[0]

    @staticmethod
    def _expected_return_percent(bullish: float, bearish: float, expected_move_percent: float) -> float:
        return (bullish - bearish) * expected_move_percent

    def _prediction_source(self, features: Mapping[str, Any]) -> str:
        """Summarize whether prediction used trained models or deterministic fallbacks."""

        source_hint = str(features.get("prediction_source") or features.get("data_source") or "").upper()
        if source_hint in {"SAMPLE", "SAMPLE_BASELINE"}:
            return "SAMPLE_BASELINE"
        if any(_is_mock_model(model) for model in self.models.values()):
            return "MOCK"
        if any(source == "trained model" for source in self._last_prediction_sources.values()):
            return "TRAINED_MODEL"
        return "FALLBACK_RULE_BASED"

    def _model_metadata(self, prediction_source: str) -> tuple[str | None, str | None]:
        """Return model name/version when a trained or mock model is used."""

        if prediction_source not in {"TRAINED_MODEL", "MOCK"}:
            return None, None
        trained_components = [name for name, source in self._last_prediction_sources.items() if source == "trained model"]
        model_names = trained_components or list(self.models.keys())
        model_name = ",".join(model_names) if model_names else None
        return model_name, self._model_version(model_name)

    def _model_version(self, model_name: str | None) -> str | None:
        if not model_name:
            return None
        try:
            entries = self.model_registry.list_models()
        except Exception:
            return None
        names = set(model_name.split(","))
        matches = [entry for entry in entries if entry.model_name in names]
        if not matches:
            return None
        latest = matches[-1]
        return str(latest.metadata.get("version") or latest.created_at)

    @staticmethod
    def _features_used_count(features: Mapping[str, Any]) -> int:
        """Count numeric/non-empty feature values made available to prediction."""

        return sum(1 for value in features.values() if _as_float(value) is not None)

    def reset_summary(self) -> None:
        """Reset prediction logging counters."""

        self._prediction_count = 0
        self._fallback_prediction_count = 0
        self._model_prediction_count = 0

    def log_summary(self) -> None:
        """Log aggregate prediction counts at INFO level."""

        if self._prediction_count <= 0:
            return
        LOGGER.info(
            "Prediction summary",
            extra={
                "predictions_generated_count": self._prediction_count,
                "fallback_predictions_count": self._fallback_prediction_count,
                "model_predictions_count": self._model_prediction_count,
            },
        )

    def _record_prediction(self, prediction_source: str) -> None:
        self._prediction_count += 1
        if prediction_source == "TRAINED_MODEL":
            self._model_prediction_count += 1
        else:
            self._fallback_prediction_count += 1


def _predict_probabilities(model: Any, features: Mapping[str, Any]) -> tuple[float, ...] | None:
    try:
        names = getattr(model, "feature_names_in_", tuple(features.keys()))
        row = [[float(features.get(name, 0.0) or 0.0) for name in names]]
        probabilities = model.predict_proba(row)[0]
        return tuple(float(value) for value in probabilities)
    except (AttributeError, TypeError, ValueError):
        return None


def _predict_scalar(model: Any, features: Mapping[str, Any]) -> float | None:
    try:
        names = getattr(model, "feature_names_in_", tuple(features.keys()))
        row = [[float(features.get(name, 0.0) or 0.0) for name in names]]
        return float(model.predict(row)[0])
    except (AttributeError, TypeError, ValueError):
        return None


def _normalize_three(values: tuple[float, ...]) -> tuple[float, float, float]:
    padded = tuple(values[:3]) + (0.0,) * max(0, 3 - len(values))
    cleaned = tuple(max(0.0, float(value)) for value in padded[:3])
    total = sum(cleaned)
    if total <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return cleaned[0] / total, cleaned[1] / total, cleaned[2] / total


def _first_float(features: Mapping[str, Any], *names: str, default: float | None = 0.0) -> float:
    for name in names:
        value = _as_float(features.get(name))
        if value is not None:
            return value
    return float(default) if default is not None else 0.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expected_expiry_distribution(
    spot: float | None,
    expected_return_percent: float,
    expected_move_percent: float,
) -> dict[str, float] | None:
    if spot is None or spot <= 0:
        return None
    return {
        "mean": spot * (1.0 + expected_return_percent / 100.0),
        "stdev": max(spot * abs(expected_move_percent) / 100.0, spot * 0.001),
    }


def _is_mock_model(model: Any) -> bool:
    name = f"{type(model).__module__}.{type(model).__name__}".lower()
    return "mock" in name or "fake" in name
