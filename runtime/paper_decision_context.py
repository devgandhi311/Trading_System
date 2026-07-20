"""Causal PAPER decision-context service.

Builds an advisory DecisionContext from validated market data, bounded
historical features, regime detection, ML prediction, volatility forecast,
expiry distribution, explainability, liquidity, and news/event-risk context.

This service never creates or submits orders, never invokes broker adapters,
and never runs risk approval or strategy selection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from features.feature_pipeline import FeaturePipeline
from ml.explainability import ExplainabilityEngine, PredictionExplanation
from ml.model_registry import ModelRegistry
from ml.prediction_engine import PredictionEngine
from models import PaperPortfolioState, PredictionResult, RegimeResult, RegimeType
from models.market_models import PaperMarketSnapshot
from models.paper_models import DecisionContext, SessionState
from news.news_config import EventRiskResult
from quant.distribution.expiry_distribution import ExpiryDistributionEngine, ExpiryDistributionResult
from quant.volatility.vol_forecaster import VolatilityForecaster, VolForecastResult
from regime.liquidity_regime import LiquidityRegimeAnalyzer
from regime.regime_detector import RegimeDetector
from storage.database import OperationalDatabase
from storage.models import DecisionCycle, TradingSession
from storage.repositories import OperationalRepository
from utils.config_loader import AppConfig, ConfigLoader
from utils.exceptions import DataValidationError
from utils.time_utils import ensure_ist, now_ist

LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURE_WINDOW = 60
MINIMUM_LIQUIDITY_SCORE = 0.35
MINIMUM_FEATURE_ROWS = 2


class PaperDecisionContextService:
    """Build and persist a causal PAPER decision context."""

    def __init__(
        self,
        repository: OperationalRepository,
        config: AppConfig | ConfigLoader | None = None,
        *,
        model_registry: ModelRegistry | None = None,
        explainability_engine: ExplainabilityEngine | None = None,
        feature_window: int = DEFAULT_FEATURE_WINDOW,
        minimum_liquidity_score: float = MINIMUM_LIQUIDITY_SCORE,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create the service with optional injected dependencies."""

        self.repository = repository
        self._config_loader = config if isinstance(config, ConfigLoader) else ConfigLoader()
        self.config = self._config_loader.load() if isinstance(config, ConfigLoader) else (config or self._config_loader.load())
        self.model_registry = model_registry
        self.explainability_engine = explainability_engine or ExplainabilityEngine()
        self.feature_window = max(1, int(feature_window))
        self.minimum_liquidity_score = float(minimum_liquidity_score)
        self._clock = clock or now_ist

        self.feature_pipeline = FeaturePipeline(mode=self.config.settings.mode)
        self.regime_detector = RegimeDetector(feature_pipeline=self.feature_pipeline)
        self.prediction_engine = PredictionEngine(
            config_loader=self._config_loader,
            model_registry=model_registry,
            explainability_engine=self.explainability_engine,
        )
        project_root = Path(__file__).resolve().parent.parent
        quant_config = self._load_yaml(project_root / "config" / "quant_config.yaml") or {}
        vol_section = quant_config.get("volatility", {})
        self.volatility_forecaster = VolatilityForecaster(
            realized_window=int(vol_section.get("realized_vol_window", 20)),
            garch_enabled=bool(vol_section.get("garch_enabled", True)),
            egarch_enabled=bool(vol_section.get("egarch_enabled", True)),
            fallback_to_realized_vol=bool(vol_section.get("fallback_to_realized_vol", True)),
        )
        dist_section = quant_config.get("distribution", {})
        self.expiry_distribution_engine = ExpiryDistributionEngine(
            confidence_levels=tuple(dist_section.get("confidence_levels", [0.68, 0.80, 0.90, 0.95])),
            use_parametric=bool(dist_section.get("use_parametric", True)),
            use_empirical=bool(dist_section.get("use_empirical", True)),
            use_monte_carlo=bool(dist_section.get("use_monte_carlo", True)),
        )
        self._vol_config = quant_config
        self.liquidity_analyzer = LiquidityRegimeAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(
        self,
        session: TradingSession,
        market_snapshot: PaperMarketSnapshot,
        historical_bars: pd.DataFrame,
        portfolio_state: PaperPortfolioState | Mapping[str, Any],
        news_state: EventRiskResult | None,
        *,
        decision_time: datetime | None = None,
    ) -> DecisionContext:
        """Build one causal advisory DecisionContext and persist it."""

        current_time = ensure_ist(decision_time or self._clock())
        invalid_reasons: list[str] = []
        warnings: list[str] = []

        cycle_id = self._cycle_id(session.id, current_time)
        self.repository.create_decision_cycle(cycle_id, session.id, started_at=current_time)

        if not self._session_permits_entries(session):
            invalid_reasons.append("Session does not permit entries")

        validation = self._validate_market_snapshot(market_snapshot, current_time)
        if not validation.valid:
            invalid_reasons.extend(validation.blocking_reasons)

        bounded_bars, causality_warnings = self._prepare_bounded_bars(historical_bars, current_time)
        warnings.extend(causality_warnings)

        if len(bounded_bars) < MINIMUM_FEATURE_ROWS:
            invalid_reasons.append(f"Insufficient history: {len(bounded_bars)} rows after bounding")

        features: pd.DataFrame | None = None
        feature_reference = None
        feature_timestamp = None
        if not invalid_reasons:
            try:
                features = self.feature_pipeline.generate_features(
                    bounded_bars,
                    rolling_window=min(self.feature_window, len(bounded_bars)),
                    cache=False,
                ).features
                if features.empty or features.isna().all().all():
                    invalid_reasons.append("Feature row is incomplete")
                else:
                    feature_timestamp = ensure_ist(features["date"].iloc[-1])
                    feature_reference = f"features:{session.id}:{feature_timestamp.isoformat()}"
            except Exception as exc:
                invalid_reasons.append(f"Feature calculation failed: {exc}")

        regime_result: RegimeResult | None = None
        regime_warnings: list[str] = []
        if not invalid_reasons and features is not None and not features.empty:
            try:
                regime_result = self.regime_detector.detect(
                    features,
                    symbol=market_snapshot.underlying,
                    news_event_risk=news_state,
                )
                if not getattr(regime_result, "allowed_to_trade", True):
                    invalid_reasons.append("Regime blocks entries")
                regime_warnings = list(getattr(regime_result, "warnings", ())) or []
            except Exception as exc:
                invalid_reasons.append(f"Regime detection failed: {exc}")

        prediction_result: PredictionResult | None = None
        prediction_warnings: list[str] = []
        if not invalid_reasons and features is not None and not features.empty:
            try:
                latest_features = features.iloc[-1].to_dict()
                prediction_result = self.prediction_engine.predict(
                    latest_features,
                    current_regime=regime_result or RegimeType.UNKNOWN,
                    option_chain_snapshot=market_snapshot.option_chain,
                    underlying=market_snapshot.underlying,
                    timestamp=current_time,
                    news_sentiment=None,
                    news_event_risk=news_state,
                )
                if not all(
                    _is_finite(v) for v in (
                        prediction_result.bullish_probability,
                        prediction_result.bearish_probability,
                        prediction_result.sideways_probability,
                        prediction_result.confidence,
                    )
                ):
                    invalid_reasons.append("Prediction output is non-finite")
                prediction_warnings = list(getattr(prediction_result, "warnings", ())) or []
            except Exception as exc:
                invalid_reasons.append(f"Prediction failed: {exc}")

        vol_result: VolForecastResult | None = None
        vol_warnings: list[str] = []
        if not invalid_reasons and features is not None and not features.empty:
            try:
                returns = self._returns_from_features(features)
                vol_result = self.volatility_forecaster.ensemble_forecast(
                    returns,
                    days_to_expiry=self._days_to_expiry(market_snapshot),
                )
                if vol_result.forecast_vol <= 0 or not _is_finite(vol_result.forecast_vol):
                    invalid_reasons.append("Volatility forecast is unusable")
                if vol_result.model_used not in {"garch", "egarch"} and vol_result.warning:
                    vol_warnings.append(vol_result.warning)
            except Exception as exc:
                invalid_reasons.append(f"Volatility forecast failed: {exc}")

        expiry_result: ExpiryDistributionResult | None = None
        expiry_warnings: list[str] = []
        if not invalid_reasons and features is not None and not features.empty:
            try:
                spot = self._spot_price(market_snapshot, features)
                expiry_result = self.expiry_distribution_engine.estimate(
                    spot_price=spot,
                    historical_returns=self._returns_from_features(features),
                    days_to_expiry=self._days_to_expiry(market_snapshot),
                    realized_volatility=vol_result.forecast_vol if vol_result else None,
                    predicted_volatility=vol_result.forecast_vol if vol_result else None,
                    regime_result=regime_result,
                    news_event_risk=news_state,
                )
                if not _is_finite(expiry_result.expected_expiry_price):
                    invalid_reasons.append("Expiry distribution is invalid")
            except Exception as exc:
                invalid_reasons.append(f"Expiry distribution failed: {exc}")

        explanation: PredictionExplanation | None = None
        if prediction_result and features is not None and not features.empty:
            try:
                explanation = self.explainability_engine.build(
                    features.iloc[-1].to_dict(),
                    confidence=prediction_result.confidence,
                    bullish_probability=prediction_result.bullish_probability,
                    bearish_probability=prediction_result.bearish_probability,
                    sideways_probability=prediction_result.sideways_probability,
                    breakout_failure_probability=getattr(prediction_result, "breakout_failure_probability", 0.0),
                    trade_allowed=not invalid_reasons,
                    strategy_type=getattr(prediction_result, "recommended_strategy_type", None),
                    regime_name=getattr(regime_result, "regime_name", None) or getattr(regime_result, "regime", RegimeType.UNKNOWN).value,
                    expected_move_percent=getattr(prediction_result, "expected_move_percent", 0.0),
                    iv_percentile=self._first_float(features.iloc[-1].to_dict(), "iv_percentile", "volatility_percentile"),
                )
                warnings.extend(explanation.warning_flags)
            except Exception as exc:
                warnings.append(f"Explainability failed: {exc}")

        liquidity_score = self._liquidity_score(features)
        if liquidity_score is not None and liquidity_score < self.minimum_liquidity_score:
            invalid_reasons.append(f"Liquidity {liquidity_score:.2f} is below minimum {self.minimum_liquidity_score:.2f}")

        if news_state is not None:
            event_level = str(getattr(news_state, "event_risk_level", "") or "").upper()
            if event_level in {"HIGH", "EXTREME"}:
                invalid_reasons.append(f"News event risk blocks entries: {event_level}")
            elif event_level == "MEDIUM":
                warnings.append("Medium news event risk reduces confidence")
        else:
            news_fail_safe = self.config.raw("news", {}).get("risk_rules", {}).get("fail_safe_block_on_news_api_error", False)
            if news_fail_safe:
                invalid_reasons.append("News provider failure follows fail-safe block")

        if portfolio_state is not None:
            equity = self._portfolio_equity(portfolio_state)
            if equity is not None and equity <= 0:
                invalid_reasons.append("Current equity is zero or negative")

        entry_context_valid = not invalid_reasons
        context = DecisionContext(
            decision_cycle_id=cycle_id,
            session_id=session.id,
            timestamp=current_time,
            market_snapshot_id=market_snapshot.snapshot_id,
            feature_snapshot_id=feature_reference,
            feature_timestamp=feature_timestamp,
            regime=getattr(regime_result, "regime", None).value if regime_result else None,
            regime_confidence=float(getattr(regime_result, "confidence", 0.0) or 0.0),
            regime_warnings=tuple(regime_warnings),
            prediction_probabilities=(
                prediction_result.bullish_probability,
                prediction_result.bearish_probability,
                prediction_result.sideways_probability,
            ) if prediction_result else None,
            prediction_confidence=float(getattr(prediction_result, "confidence", 0.0) or 0.0),
            expected_return_percent=getattr(prediction_result, "expected_return_percent", None),
            expected_range_low=getattr(prediction_result, "expected_range_low", None),
            expected_range_high=getattr(prediction_result, "expected_range_high", None),
            model_names=tuple(self._model_names(prediction_result)),
            model_versions=tuple(self._model_versions(prediction_result)),
            volatility_forecast=vol_result.forecast_vol_percent if vol_result else None,
            volatility_model_requested="garch" if self._vol_config.get("volatility", {}).get("garch_enabled") else "realized_vol",
            volatility_model_used=vol_result.model_used if vol_result else None,
            convergence_status="converged" if (vol_result and vol_result.model_used in {"garch", "egarch"}) else ("fallback" if vol_result else "unavailable"),
            fallback_used=bool(vol_warnings) or bool(getattr(prediction_result, "prediction_source", "") != "TRAINED_MODEL"),
            fallback_reason="; ".join(vol_warnings) if vol_warnings else None,
            expected_expiry_price=expiry_result.expected_expiry_price if expiry_result else None,
            expected_move=expiry_result.expected_move_points if expiry_result else None,
            range_68=(expiry_result.lower_bound_68, expiry_result.upper_bound_68) if expiry_result else None,
            range_95=(expiry_result.lower_bound_95, expiry_result.upper_bound_95) if expiry_result else None,
            tail_risk_score=expiry_result.tail_risk_score if expiry_result else None,
            news_sentiment=getattr(news_state, "event_risk_score", None) if news_state else None,
            event_risk_level=str(getattr(news_state, "event_risk_level", "") or "").upper() if news_state else None,
            liquidity_score=liquidity_score,
            entry_context_valid=entry_context_valid,
            invalid_reason="; ".join(invalid_reasons) if invalid_reasons else None,
            warnings=tuple(warnings),
            metadata={
                "causality_window": self.feature_window,
                "market_snapshot_timestamp": market_snapshot.timestamp.isoformat(),
                "decision_time": current_time.isoformat(),
                "feature_count": len(features) if features is not None else 0,
                "explainability": explanation.as_dict() if explanation else {},
                "model_source": getattr(prediction_result, "prediction_source", None),
            },
        )

        self.repository.save_decision_context(context)
        self.repository.complete_decision_cycle(cycle_id, completed_at=current_time)
        return context

    # ------------------------------------------------------------------
    # Input validation and causality guards
    # ------------------------------------------------------------------

    def _validate_market_snapshot(self, snapshot: PaperMarketSnapshot, decision_time: datetime) -> Any:
        from runtime.market_snapshot_validator import MarketSnapshotValidator, MarketSnapshotValidationPolicy
        validator = MarketSnapshotValidator(policy=MarketSnapshotValidationPolicy())
        return validator.validate(snapshot, current_time=decision_time)

    def _prepare_bounded_bars(self, bars: pd.DataFrame, decision_time: datetime) -> tuple[pd.DataFrame, list[str]]:
        warnings: list[str] = []
        if bars is None or bars.empty:
            return pd.DataFrame(), warnings

        data = bars.copy()
        if "date" not in data.columns and "timestamp" not in data.columns:
            raise DataValidationError("Historical bars must contain 'date' or 'timestamp' column")

        time_col = "date" if "date" in data.columns else "timestamp"
        data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
        data = data.dropna(subset=[time_col])

        future_count = int((data[time_col] > decision_time).sum())
        if future_count:
            warnings.append(f"Discarded {future_count} future rows after bounding")

        data = data[data[time_col] <= decision_time]
        data = data.sort_values(time_col)
        data = data.drop_duplicates(subset=[time_col], keep="last")

        if len(data) > self.feature_window:
            data = data.iloc[-self.feature_window:]
            warnings.append(f"Historical bars bounded to last {self.feature_window} rows")

        return data.reset_index(drop=True), warnings

    def _session_permits_entries(self, session: TradingSession) -> bool:
        try:
            state = SessionState(session.status)
            return state == SessionState.ENTRY_ALLOWED
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Feature and model helpers
    # ------------------------------------------------------------------

    def _returns_from_features(self, features: pd.DataFrame) -> list[float]:
        for column in ("daily_return", "log_return", "intraday_return"):
            if column in features.columns:
                values = features[column].dropna().tolist()
                if values:
                    return [float(v) for v in values]
        close = features.get("close")
        if close is not None and len(close) > 1:
            returns = close.pct_change().dropna().tolist()
            if returns:
                return [float(v) for v in returns]
        return [0.0]

    def _spot_price(self, snapshot: PaperMarketSnapshot, features: pd.DataFrame | None) -> float:
        if snapshot.spot_price and snapshot.spot_price > 0:
            return float(snapshot.spot_price)
        if features is not None and not features.empty:
            close = self._first_float_series(features, "close", "spot_price", "last_price")
            if close is not None:
                return close
        raise DataValidationError("Missing spot price from snapshot and features")

    def _days_to_expiry(self, snapshot: PaperMarketSnapshot) -> int:
        expiry = getattr(snapshot.option_chain, "expiry", None)
        if expiry is None:
            return 1
        now = ensure_ist(self._clock()).date()
        delta = (expiry - now).days
        return max(1, delta)

    def _liquidity_score(self, features: pd.DataFrame | None) -> float | None:
        if features is None or features.empty:
            return None
        for column in ("liquidity_score", "liquidity_regime_score"):
            if column in features.columns:
                value = features[column].iloc[-1]
                if _is_finite(value):
                    return float(value)
        try:
            return float(self.liquidity_analyzer.liquidity_score(features))
        except Exception:
            return None

    def _model_names(self, prediction: PredictionResult | None) -> tuple[str, ...]:
        if prediction is None:
            return ()
        name = getattr(prediction, "model_name", None)
        if not name:
            return ()
        return tuple(str(item).strip() for item in str(name).split(",") if str(item).strip())

    def _model_versions(self, prediction: PredictionResult | None) -> tuple[str, ...]:
        if prediction is None:
            return ()
        version = getattr(prediction, "model_version", None)
        if not version:
            return ()
        return (str(version),)

    def _portfolio_equity(self, portfolio_state: PaperPortfolioState | Mapping[str, Any]) -> float | None:
        if isinstance(portfolio_state, PaperPortfolioState):
            return float(portfolio_state.current_equity)
        if isinstance(portfolio_state, Mapping):
            return self._first_float_mapping(portfolio_state, "current_equity", "current_equity")
        return None

    # ------------------------------------------------------------------
    # Static / utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data if isinstance(data, dict) else None

    @staticmethod
    def _first_float_series(frame: pd.DataFrame, *names: str) -> float | None:
        for name in names:
            if name in frame.columns:
                value = frame[name].iloc[-1]
                if _is_finite(value):
                    return float(value)
        return None

    @staticmethod
    def _first_float_mapping(mapping: Mapping[str, Any], *names: str) -> float | None:
        for name in names:
            value = mapping.get(name)
            if _is_finite(value):
                return float(value)
        return None

    @staticmethod
    def _cycle_id(session_id: str, timestamp: datetime) -> str:
        return f"cycle-{session_id}-{timestamp.strftime('%Y%m%d%H%M%S%f')}"

    @staticmethod
    def _first_float(features: Mapping[str, Any], *names: str, default: float | None = None) -> float | None:
        for name in names:
            value = features.get(name)
            if _is_finite(value):
                return float(value)
        return float(default) if default is not None else None


def _is_finite(value: Any) -> bool:
    try:
        return bool(float(value) == float(value) and float(value) not in (float("inf"), float("-inf")))
    except (TypeError, ValueError):
        return False
