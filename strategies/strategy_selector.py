"""Strategy selection based on prediction, regime, option, and risk context."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from models import OptionChainSnapshot, PredictionResult, StrategyLeg, StrategyPlan, StrategyType
from strategies.base_strategy import BaseStrategy
from strategies.expected_move_credit_spread import ExpectedMoveCreditSpreadStrategy
from utils.config_loader import ConfigLoader
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)


class StrategySelector:
    """Choose a StrategyPlan without placing orders."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        strategies: Mapping[StrategyType, BaseStrategy] | None = None,
    ) -> None:
        """Create a selector from app configuration."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.strategies = dict(strategies or self._default_strategies())

    def select(
        self,
        prediction: PredictionResult,
        option_chain_snapshot: OptionChainSnapshot | None = None,
        *,
        features: Mapping[str, Any] | None = None,
        risk_context: Mapping[str, Any] | None = None,
        time_to_expiry_days: int | None = None,
    ) -> StrategyPlan:
        """Return the best strategy plan or a blocked empty plan."""

        feature_map = dict(features or {})
        risk_map = dict(risk_context or {})
        allowed = {StrategyType(item) for item in self.config.strategy.allowed_strategies if item in StrategyType._value2member_map_}
        candidate_type = self._candidate_strategy(prediction, feature_map)

        if candidate_type is None:
            return self._blocked_plan(prediction, "Prediction did not suggest a strategy.")
        if candidate_type not in allowed:
            return self._blocked_plan(prediction, f"{candidate_type.value} is not allowed by strategy config.")
        if not prediction.trade_allowed:
            return self._blocked_plan(prediction, prediction.reason or "Prediction blocked trading.")

        iv_percentile = _as_float(feature_map.get("iv_percentile"))
        liquidity_score = _as_float(feature_map.get("liquidity_score"))
        liquidity_score = 0.6 if liquidity_score is None else liquidity_score
        expected_range = self._expected_range(prediction)
        expiry_days = self._time_to_expiry_days(option_chain_snapshot, time_to_expiry_days)
        filter_ok, filter_reason = self._passes_filters(
            candidate_type,
            prediction=prediction,
            expected_range=expected_range,
            iv_percentile=iv_percentile,
            liquidity_score=liquidity_score,
            risk_context=risk_map,
            time_to_expiry_days=expiry_days,
        )
        if not filter_ok:
            return self._blocked_plan(prediction, filter_reason)

        strategy = self.strategies[candidate_type]
        lot_size = self.config.strategy.lot_size.get((prediction.underlying or prediction.symbol).upper(), 1)
        plan = strategy.build_plan(
            prediction,
            option_chain_snapshot,
            lot_size=lot_size,
            max_lots=self.config.strategy.max_lots_per_trade,
            liquidity_score=liquidity_score,
        )
        LOGGER.info(
            "Strategy selected",
            extra={
                "strategy_type": plan.strategy_type.value,
                "symbol": plan.symbol,
                "valid": plan.metadata.get("valid"),
            },
        )
        return plan

    def _candidate_strategy(
        self,
        prediction: PredictionResult,
        features: Mapping[str, Any],
    ) -> StrategyType | None:
        if prediction.recommended_strategy_type:
            try:
                return StrategyType(prediction.recommended_strategy_type)
            except ValueError:
                return None
        if prediction.sideways_probability >= max(prediction.bullish_probability, prediction.bearish_probability):
            if StrategyType.EXPECTED_MOVE_CREDIT_SPREAD in self.strategies:
                return StrategyType.EXPECTED_MOVE_CREDIT_SPREAD
            return StrategyType.MEAN_REVERSION
        if (
            _as_float(features.get("iv_percentile") or 0.0) >= 55.0
            and prediction.expected_move_percent > 0
            and StrategyType.EXPECTED_MOVE_CREDIT_SPREAD in self.strategies
        ):
            return StrategyType.EXPECTED_MOVE_CREDIT_SPREAD
        if prediction.volatility_state == "expanding" or _as_float(features.get("iv_percentile") or 0.0) >= 70.0:
            return StrategyType.VOLATILITY_SCALPING
        if max(prediction.bullish_probability, prediction.bearish_probability) >= 0.45:
            return StrategyType.MOMENTUM_BREAKOUT
        return None

    def _passes_filters(
        self,
        strategy_type: StrategyType,
        *,
        prediction: PredictionResult,
        expected_range: float,
        iv_percentile: float | None,
        liquidity_score: float,
        risk_context: Mapping[str, Any],
        time_to_expiry_days: int,
    ) -> tuple[bool, str]:
        if prediction.confidence_score < self.config.strategy.min_confidence_score:
            return False, "Prediction confidence is below configured minimum."
        if expected_range <= 0:
            return False, "Expected range is unavailable."
        if liquidity_score < 0.35:
            return False, "Liquidity score is below strategy threshold."
        max_loss_allowed = _as_float(risk_context.get("max_loss_allowed"))
        estimated_loss = _as_float(risk_context.get("estimated_loss"))
        if max_loss_allowed is not None and estimated_loss is not None and estimated_loss > max_loss_allowed:
            return False, "Estimated loss exceeds risk limit."
        if time_to_expiry_days < 0:
            return False, "Option expiry is already past."
        if strategy_type == StrategyType.VOLATILITY_SCALPING and iv_percentile is not None and iv_percentile < 45.0:
            return False, "IV percentile is too low for volatility scalping."
        if strategy_type == StrategyType.EXPECTED_MOVE_CREDIT_SPREAD:
            params = self.config.strategy.expected_move_credit_spread
            min_iv = _as_float(params.get("min_iv_percentile"))
            if min_iv is not None and iv_percentile is not None and iv_percentile < min_iv:
                return False, "IV percentile is too low for expected-move credit spread."
            news_risk = str(risk_context.get("event_risk_level") or prediction.metadata.get("event_risk_level") or "").upper()
            if news_risk in {"HIGH", "EXTREME"}:
                return False, f"News event risk {news_risk} blocks expected-move credit spread."
        if strategy_type == StrategyType.MEAN_REVERSION and iv_percentile is not None and iv_percentile > 85.0:
            return False, "IV percentile is too high for mean reversion."
        if prediction.breakout_failure_probability > 0.65 and strategy_type == StrategyType.MOMENTUM_BREAKOUT:
            return False, "Breakout failure probability is too high for momentum breakout."
        return True, "Strategy filters passed."

    def _blocked_plan(self, prediction: PredictionResult, reason: str) -> StrategyPlan:
        return StrategyPlan(
            strategy_id=f"blocked-{now_ist().strftime('%Y%m%d%H%M%S')}",
            strategy_type=StrategyType.CUSTOM,
            symbol=prediction.underlying or prediction.symbol,
            created_at=now_ist(),
            legs=(),
            confidence=prediction.confidence_score,
            rationale=reason,
            max_loss=0.0,
            expected_profit=0.0,
            metadata={"valid": False, "blocked": True, "validation_reason": reason, "advisory_only": True},
        )

    def _default_strategies(self) -> dict[StrategyType, BaseStrategy]:
        allowed_regimes = tuple(self.config.strategy.allowed_regimes)
        required_confidence = self.config.strategy.min_confidence_score
        expected_move_params = self.config.strategy.expected_move_credit_spread
        return {
            StrategyType.MOMENTUM_BREAKOUT: BaseStrategy(
                name="Momentum Breakout",
                strategy_type=StrategyType.MOMENTUM_BREAKOUT,
                allowed_regimes=allowed_regimes,
                forbidden_regimes=("range_bound",),
                required_confidence=required_confidence,
            ),
            StrategyType.MEAN_REVERSION: BaseStrategy(
                name="Mean Reversion",
                strategy_type=StrategyType.MEAN_REVERSION,
                allowed_regimes=allowed_regimes,
                forbidden_regimes=("high_volatility",),
                required_confidence=required_confidence,
            ),
            StrategyType.VOLATILITY_SCALPING: BaseStrategy(
                name="Volatility Scalping",
                strategy_type=StrategyType.VOLATILITY_SCALPING,
                allowed_regimes=allowed_regimes,
                required_confidence=required_confidence,
            ),
            StrategyType.EXPECTED_MOVE_CREDIT_SPREAD: ExpectedMoveCreditSpreadStrategy(
                allowed_regimes=allowed_regimes,
                required_confidence=required_confidence,
                min_probability_otm=_config_float(expected_move_params, "min_probability_otm", 0.68),
                min_reward_risk=_config_float(expected_move_params, "min_reward_risk", 0.20),
                max_spread_width=_config_float(expected_move_params, "max_spread_width", 500.0),
                max_bid_ask_spread_percent=_config_float(expected_move_params, "max_bid_ask_spread_percent", 5.0),
                confidence_interval_z=_config_float(expected_move_params, "confidence_interval_z", 1.0),
                expected_shortfall_tail_percent=_config_float(
                    expected_move_params,
                    "expected_shortfall_tail_percent",
                    5.0,
                ),
                min_iv_percentile=_config_float(expected_move_params, "min_iv_percentile", 35.0),
            ),
        }

    @staticmethod
    def _expected_range(prediction: PredictionResult) -> float:
        if prediction.expected_range_low is None or prediction.expected_range_high is None:
            return prediction.expected_move_percent
        return max(0.0, prediction.expected_range_high - prediction.expected_range_low)

    @staticmethod
    def _time_to_expiry_days(snapshot: OptionChainSnapshot | None, override: int | None) -> int:
        if override is not None:
            return override
        if snapshot is None:
            return 7
        return max(0, (snapshot.expiry - snapshot.timestamp.date()).days)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = _as_float(config.get(key))
    return default if value is None else value
