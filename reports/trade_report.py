"""Trade-level reporting schema and row builders."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Mapping

from models import PredictionResult, RegimeResult, RiskDecision, StrategyPlan


TRADE_REPORT_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "date",
    "entry_time",
    "exit_time",
    "underlying",
    "expiry",
    "strategy",
    "legs",
    "quantity",
    "entry_price",
    "exit_price",
    "trigger_price",
    "limit_price",
    "order_type",
    "fill_status",
    "gross_pnl",
    "net_pnl",
    "brokerage_placeholder",
    "slippage",
    "max_adverse_excursion",
    "max_favorable_excursion",
    "regime_at_entry",
    "regime_at_exit",
    "ml_confidence",
    "prediction_source",
    "model_name",
    "model_version",
    "features_used_count",
    "direction_probability",
    "expected_move",
    "expected_value",
    "probability_of_profit",
    "expected_shortfall",
    "risk_allowed",
    "risk_used",
    "greeks",
    "iv",
    "iv_percentile",
    "oi_change",
    "bid_ask_spread",
    "entry_reason",
    "exit_reason",
    "mistake_flag",
    "improvement_note",
    "latency_breakdown",
    "news_sentiment_score",
    "event_risk_level",
    "top_relevant_headline",
    "news_reason",
)


class TradeReportBuilder:
    """Build normalized trade-report rows."""

    columns = TRADE_REPORT_COLUMNS

    def build_row(
        self,
        trade: Mapping[str, Any] | Any,
        *,
        strategy_plan: StrategyPlan | None = None,
        prediction: PredictionResult | None = None,
        risk_decision: RiskDecision | None = None,
        entry_regime: RegimeResult | str | None = None,
        exit_regime: RegimeResult | str | None = None,
        latency_breakdown: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one complete trade-report row."""

        data = _to_mapping(trade)
        metadata = _metadata(data)
        plan_metadata = strategy_plan.metadata if strategy_plan else {}
        row = {
            "trade_id": _first(data, "trade_id", "order_id", "strategy_id", default=strategy_plan.strategy_id if strategy_plan else ""),
            "date": _date_value(_first(data, "date", "entry_date", "entry_time", default=None)),
            "entry_time": _iso(_first(data, "entry_time", "entry_date", default=None)),
            "exit_time": _iso(_first(data, "exit_time", "exit_date", default=None)),
            "underlying": _first(data, "underlying", "symbol", default=strategy_plan.symbol if strategy_plan else ""),
            "expiry": _first(data, "expiry", default=metadata.get("expiry", plan_metadata.get("expiry_type", ""))),
            "strategy": _first(data, "strategy", "strategy_type", default=strategy_plan.strategy_type.value if strategy_plan else ""),
            "legs": _stringify(_first(data, "legs", default=_plan_legs(strategy_plan))),
            "quantity": _first(data, "quantity", default=_plan_quantity(strategy_plan)),
            "entry_price": _first(data, "entry_price", "average_price", default=""),
            "exit_price": _first(data, "exit_price", default=""),
            "trigger_price": _first(data, "trigger_price", default=metadata.get("trigger_price", "")),
            "limit_price": _first(data, "limit_price", default=metadata.get("limit_price", "")),
            "order_type": _first(data, "order_type", default=metadata.get("order_type", "")),
            "fill_status": _first(data, "fill_status", "status", default=metadata.get("fill_status", "")),
            "gross_pnl": _first(data, "gross_pnl", default=0.0),
            "net_pnl": _first(data, "net_pnl", "realized_pnl", default=0.0),
            "brokerage_placeholder": _first(data, "brokerage_placeholder", "brokerage", default=metadata.get("brokerage", 0.0)),
            "slippage": _first(data, "slippage", default=metadata.get("slippage", metadata.get("slippage_percent", 0.0))),
            "max_adverse_excursion": _first(data, "max_adverse_excursion", "mae", default=metadata.get("max_adverse_excursion", 0.0)),
            "max_favorable_excursion": _first(data, "max_favorable_excursion", "mfe", default=metadata.get("max_favorable_excursion", 0.0)),
            "regime_at_entry": _regime_name(entry_regime) or _first(data, "regime_at_entry", "regime", default=""),
            "regime_at_exit": _regime_name(exit_regime) or _first(data, "regime_at_exit", default=""),
            "ml_confidence": _first(data, "ml_confidence", default=prediction.confidence_score if prediction else ""),
            "prediction_source": _first(data, "prediction_source", default=prediction.prediction_source if prediction else ""),
            "model_name": _first(data, "model_name", default=prediction.model_name if prediction else ""),
            "model_version": _first(data, "model_version", default=prediction.model_version if prediction else ""),
            "features_used_count": _first(data, "features_used_count", default=prediction.features_used_count if prediction else ""),
            "direction_probability": _first(data, "direction_probability", default=_direction_probability(prediction)),
            "expected_move": _first(data, "expected_move", default=prediction.expected_move_percent if prediction else ""),
            "expected_value": _first(data, "expected_value", default=metadata.get("expected_value", plan_metadata.get("expected_value", ""))),
            "probability_of_profit": _first(
                data,
                "probability_of_profit",
                default=metadata.get("probability_of_profit", plan_metadata.get("probability_of_profit", "")),
            ),
            "expected_shortfall": _first(
                data,
                "expected_shortfall",
                default=metadata.get("expected_shortfall", plan_metadata.get("expected_shortfall", "")),
            ),
            "risk_allowed": _first(data, "risk_allowed", default=risk_decision.approved if risk_decision else ""),
            "risk_used": _first(data, "risk_used", default=risk_decision.risk_used if risk_decision else ""),
            "greeks": _stringify(_first(data, "greeks", default=metadata.get("greeks", {}))),
            "iv": _first(data, "iv", default=metadata.get("iv", metadata.get("implied_volatility", ""))),
            "iv_percentile": _first(data, "iv_percentile", default=metadata.get("iv_percentile", "")),
            "oi_change": _first(data, "oi_change", default=metadata.get("oi_change", "")),
            "bid_ask_spread": _first(data, "bid_ask_spread", default=metadata.get("bid_ask_spread", metadata.get("bid_ask_spread_percent", ""))),
            "entry_reason": _first(data, "entry_reason", default=metadata.get("entry_reason", strategy_plan.rationale if strategy_plan else "")),
            "exit_reason": _first(data, "exit_reason", default=metadata.get("exit_reason", "")),
            "mistake_flag": _first(data, "mistake_flag", default=metadata.get("mistake_flag", False)),
            "improvement_note": _first(data, "improvement_note", default=metadata.get("improvement_note", "")),
            "latency_breakdown": _stringify(_first(data, "latency_breakdown", default=latency_breakdown or metadata.get("latency_breakdown", {}))),
            "news_sentiment_score": _first(data, "news_sentiment_score", default=metadata.get("news_sentiment_score", "")),
            "event_risk_level": _first(data, "event_risk_level", default=metadata.get("event_risk_level", "")),
            "top_relevant_headline": _first(data, "top_relevant_headline", default=metadata.get("top_relevant_headline", "")),
            "news_reason": _first(data, "news_reason", default=metadata.get("news_reason", "")),
        }
        return {column: row.get(column, "") for column in self.columns}

    def build_rows(self, trades: list[Mapping[str, Any] | Any]) -> list[dict[str, Any]]:
        """Build report rows for many trades."""

        return [self.build_row(trade) for trade in trades]


def _to_mapping(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported trade row type: {type(value).__name__}")


def _metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    value = data.get("metadata", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _first(data: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _date_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return datetime.fromisoformat(str(value)).date().isoformat()
    except ValueError:
        return str(value)


def _iso(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _plan_legs(plan: StrategyPlan | None) -> list[dict[str, Any]]:
    if plan is None:
        return []
    return [
        {
            "symbol": leg.symbol,
            "side": leg.side.value,
            "quantity": leg.quantity,
            "order_type": leg.order_type.value,
            "limit_price": leg.limit_price,
            "trigger_price": leg.trigger_price,
        }
        for leg in plan.legs
    ]


def _plan_quantity(plan: StrategyPlan | None) -> int | str:
    if plan is None:
        return ""
    return sum(leg.quantity for leg in plan.legs)


def _regime_name(regime: RegimeResult | str | None) -> str:
    if regime is None:
        return ""
    if isinstance(regime, str):
        return regime
    return regime.regime_name or regime.regime.value


def _direction_probability(prediction: PredictionResult | None) -> str:
    if prediction is None:
        return ""
    return str(
        {
            "bullish": prediction.bullish_probability,
            "bearish": prediction.bearish_probability,
            "sideways": prediction.sideways_probability,
        }
    )
