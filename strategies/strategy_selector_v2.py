"""Research-only strategy selector V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from strategies.broken_wing_butterfly import BrokenWingButterflyResearchStrategy
from strategies.calendar_diagonal import CalendarSpreadStrategy, DiagonalSpreadStrategy
from strategies.expected_move_credit_spread import ExpectedMoveCreditSpreadResearchStrategy, _read_value, _regime_name
from strategies.expected_move_iron_condor import ExpectedMoveIronCondorResearchStrategy
from strategies.iv_mean_reversion import IVMeanReversionPremiumResearchStrategy
from strategies.strategy_research_base import StrategyResearchResult
from strategies.volatility_expansion import VolatilityExpansionResearchStrategy


@dataclass(frozen=True)
class StrategySelectionV2Result:
    """Research-only StrategySelectorV2 output."""

    selected_strategy: str
    selected_candidate: StrategyResearchResult | None
    all_candidates_summary: tuple[dict[str, Any], ...]
    rejection_reasons: tuple[str, ...]
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


class StrategySelectorV2:
    """Select the best research candidate across advanced strategy modules."""

    def __init__(self, *, allow_calendar_diagonal_placeholders: bool = True) -> None:
        """Create selector with research-only strategy instances."""

        self.credit_spread = ExpectedMoveCreditSpreadResearchStrategy(max_bid_ask_spread_percent=15.0)
        self.iron_condor = ExpectedMoveIronCondorResearchStrategy(max_bid_ask_spread_percent=15.0)
        self.broken_wing = BrokenWingButterflyResearchStrategy(max_bid_ask_spread_percent=12.0)
        self.iv_mean_reversion = IVMeanReversionPremiumResearchStrategy()
        self.vol_expansion = VolatilityExpansionResearchStrategy()
        self.calendar = CalendarSpreadStrategy()
        self.diagonal = DiagonalSpreadStrategy()
        self.allow_calendar_diagonal_placeholders = allow_calendar_diagonal_placeholders

    def select(
        self,
        *,
        current_regime: Any,
        prediction_result: Any,
        expiry_distribution: Any,
        volatility_forecast: Any,
        option_chain: Any,
        news_risk: Any | None = None,
        risk_limits: Any | None = None,
        advanced_regime_result: Any | None = None,
    ) -> StrategySelectionV2Result:
        """Generate candidates from all eligible research strategies and select the best."""

        context = self._context(
            current_regime=current_regime,
            prediction_result=prediction_result,
            expiry_distribution=expiry_distribution,
            volatility_forecast=volatility_forecast,
            option_chain=option_chain,
            news_risk=news_risk,
            risk_limits=risk_limits,
            advanced_regime_result=advanced_regime_result,
        )
        candidates = self._generate_all(context)
        scored = [self._rescore(candidate, context) for candidate in candidates if candidate is not None]
        accepted = [candidate for candidate in scored if candidate.reject_reason is None]
        summaries = tuple(_summary(candidate) for candidate in scored)
        rejections = tuple(
            f"{candidate.strategy_name}: {candidate.reject_reason}"
            for candidate in scored
            if candidate.reject_reason
        )
        if not accepted:
            return StrategySelectionV2Result(
                selected_strategy="NO_TRADE",
                selected_candidate=None,
                all_candidates_summary=summaries,
                rejection_reasons=rejections,
                explanation="NO_TRADE: no candidate passed research filters.",
                metadata=context,
            )
        selected = max(accepted, key=lambda candidate: _selector_score(candidate, context))
        return StrategySelectionV2Result(
            selected_strategy=selected.strategy_name,
            selected_candidate=selected,
            all_candidates_summary=summaries,
            rejection_reasons=rejections,
            explanation=f"Selected {selected.strategy_name} with score {_selector_score(selected, context):.3f}.",
            metadata=context,
        )

    def _generate_all(self, context: Mapping[str, Any]) -> list[StrategyResearchResult | None]:
        option_chain = context["option_chain"]
        distribution = context["expiry_distribution"]
        regime = context["current_regime"]
        news_risk = context.get("news_risk")
        risk_limits = context.get("risk_limits")
        iv_percentile = context["iv_percentile"]
        liquidity_score = context["liquidity_score"]
        forecast_vol = context["forecast_volatility"]
        realized_vol = context["realized_volatility"]
        breakout_probability = context["breakout_probability"]
        candidates: list[StrategyResearchResult | None] = [
            self.credit_spread.analyze(
                option_chain_snapshot=option_chain,
                expiry_distribution=distribution,
                regime=regime,
                news_risk=news_risk,
                iv_percentile=iv_percentile,
                liquidity_score=liquidity_score,
                risk_limits=risk_limits,
            ),
            self.iron_condor.analyze(
                option_chain_snapshot=option_chain,
                expiry_distribution=distribution,
                regime=regime,
                news_risk=news_risk,
                iv_percentile=iv_percentile,
                liquidity_score=liquidity_score,
                risk_limits=risk_limits,
            ),
            self.broken_wing.analyze(
                option_chain_snapshot=option_chain,
                expiry_distribution=distribution,
                regime=regime,
                news_risk=news_risk,
                iv_percentile=iv_percentile,
                liquidity_score=liquidity_score,
                risk_limits=risk_limits,
            ),
            self.iv_mean_reversion.analyze(
                option_chain_snapshot=option_chain,
                expiry_distribution=distribution,
                regime=regime,
                news_risk=news_risk,
                iv_percentile=iv_percentile,
                implied_volatility=context["implied_volatility"],
                forecast_volatility=forecast_vol,
                liquidity_score=liquidity_score,
                risk_limits=risk_limits,
                major_event_nearby=context["major_event_nearby"],
            ),
            self.vol_expansion.analyze(
                option_chain_snapshot=option_chain,
                expiry_distribution=distribution,
                realized_volatility=realized_vol,
                forecast_volatility=forecast_vol,
                breakout_probability=breakout_probability,
                news_risk=news_risk,
                iv_percentile=iv_percentile,
                risk_limits=risk_limits,
            ),
        ]
        if self.allow_calendar_diagonal_placeholders:
            candidates.extend(
                [
                    self.calendar.analyze(
                        near_expiry_option_chain=option_chain,
                        far_expiry_option_chain=None,
                        iv_term_structure=None,
                        expected_move=_read_value(distribution, "expected_move_points"),
                        regime=regime,
                    ),
                    self.diagonal.analyze(
                        near_expiry_option_chain=option_chain,
                        far_expiry_option_chain=None,
                        iv_term_structure=None,
                        expected_move=_read_value(distribution, "expected_move_points"),
                        regime=regime,
                    ),
                ]
            )
        return candidates

    @staticmethod
    def _context(**kwargs: Any) -> dict[str, Any]:
        prediction = kwargs["prediction_result"]
        volatility_forecast = kwargs["volatility_forecast"]
        advanced = kwargs.get("advanced_regime_result")
        return {
            **kwargs,
            "iv_percentile": float(_read_value(prediction, "metadata") and _read_value(prediction, "metadata").get("iv_percentile") or getattr(prediction, "iv_percentile", 60.0) or 60.0),
            "liquidity_score": float(_read_value(prediction, "metadata") and _read_value(prediction, "metadata").get("liquidity_score") or 0.7),
            "forecast_volatility": float(_read_value(volatility_forecast, "forecast_vol") or _read_value(volatility_forecast, "forecast_vol_percent") or 0.20),
            "realized_volatility": float(_read_value(prediction, "metadata") and _read_value(prediction, "metadata").get("realized_volatility") or 0.12),
            "implied_volatility": float(_read_value(prediction, "metadata") and _read_value(prediction, "metadata").get("implied_volatility") or 0.25),
            "breakout_probability": float(_read_value(prediction, "breakout_failure_probability") or 0.6),
            "major_event_nearby": str(_read_value(kwargs.get("news_risk"), "event_risk_level") or "").upper() in {"HIGH", "EXTREME"},
            "advanced_trade_allowed": bool(getattr(advanced, "trade_allowed", True)),
            "regime_name": _regime_name(kwargs["current_regime"]),
        }

    @staticmethod
    def _rescore(candidate: StrategyResearchResult, context: Mapping[str, Any]) -> StrategyResearchResult:
        if candidate.reject_reason:
            return candidate
        news_level = str(_read_value(context.get("news_risk"), "event_risk_level") or "LOW").upper()
        if news_level in {"HIGH", "EXTREME"} and "VolatilityExpansion" not in candidate.strategy_name:
            return _with_reject(candidate, f"News risk {news_level} blocks candidate.")
        if not context.get("advanced_trade_allowed", True):
            return _with_reject(candidate, "Advanced regime layer blocks trade.")
        return candidate


def _selector_score(candidate: StrategyResearchResult, context: Mapping[str, Any]) -> float:
    max_loss_penalty = min(candidate.max_loss / max(_max_risk_amount(context.get("risk_limits")), 1.0), 2.0)
    drawdown_estimate = candidate.tail_loss / max(candidate.max_loss, 1.0)
    liquidity = float(candidate.metadata.get("liquidity_score", context.get("liquidity_score", 0.5)))
    news_penalty = {"LOW": 0.0, "MEDIUM": 0.10, "HIGH": 0.35, "EXTREME": 0.75}.get(
        str(_read_value(context.get("news_risk"), "event_risk_level") or "LOW").upper(),
        0.0,
    )
    regime_fit = 0.15 if _regime_name(context.get("current_regime")).lower() in str(candidate.metadata.get("regime", "")).lower() else 0.05
    volatility_fit = min(max(float(context.get("iv_percentile", 50.0)) / 100.0, 0.0), 1.0) * 0.10
    return (
        candidate.expected_value / max(candidate.max_loss, 1.0)
        + candidate.probability_of_profit * 0.35
        + candidate.risk_reward * 0.10
        + liquidity * 0.15
        + regime_fit
        + volatility_fit
        - max_loss_penalty * 0.10
        - drawdown_estimate * 0.15
        - news_penalty
    )


def _summary(candidate: StrategyResearchResult) -> dict[str, Any]:
    return {
        "strategy_name": candidate.strategy_name,
        "expected_value": candidate.expected_value,
        "probability_of_profit": candidate.probability_of_profit,
        "max_loss": candidate.max_loss,
        "tail_loss": candidate.tail_loss,
        "risk_reward": candidate.risk_reward,
        "confidence": candidate.confidence,
        "reject_reason": candidate.reject_reason,
    }


def _max_risk_amount(risk_limits: Any) -> float:
    if risk_limits is None:
        return 100_000.0
    if isinstance(risk_limits, Mapping):
        if "max_loss_amount" in risk_limits:
            return float(risk_limits["max_loss_amount"])
        if "capital" in risk_limits and "max_loss_per_trade_percent" in risk_limits:
            return float(risk_limits["capital"]) * float(risk_limits["max_loss_per_trade_percent"]) / 100.0
    return float(getattr(risk_limits, "max_loss_amount", 100_000.0))


def _with_reject(candidate: StrategyResearchResult, reason: str) -> StrategyResearchResult:
    return StrategyResearchResult(
        candidate.strategy_name,
        candidate.legs,
        candidate.expected_value,
        candidate.probability_of_profit,
        candidate.max_profit,
        candidate.max_loss,
        candidate.breakevens,
        candidate.payoff_curve,
        candidate.tail_loss,
        candidate.risk_reward,
        candidate.confidence,
        reason,
        candidate.explanation,
        candidate.metadata,
    )
