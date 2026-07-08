"""Bid-ask spread and slippage risk model."""

from __future__ import annotations

from dataclasses import dataclass

from models import StrategyPlan
from utils.math_utils import safe_divide


@dataclass(frozen=True)
class SlippageResult:
    """Result of spread and slippage checks."""

    approved: bool
    reason: str
    warnings: tuple[str, ...]
    max_spread_percent: float
    estimated_slippage_percent: float


class SlippageModel:
    """Estimate and validate transaction cost risk."""

    def __init__(self, *, max_bid_ask_spread_percent: float, max_slippage_percent: float) -> None:
        """Create spread and slippage guard."""

        self.max_bid_ask_spread_percent = float(max_bid_ask_spread_percent)
        self.max_slippage_percent = float(max_slippage_percent)

    def check(
        self,
        plan: StrategyPlan,
        *,
        market_context: dict | None = None,
    ) -> SlippageResult:
        """Validate bid-ask spread and slippage estimates."""

        context = market_context or {}
        max_spread = self._max_spread_percent(plan, context)
        slippage = self._slippage_percent(plan, context, max_spread)
        if max_spread > self.max_bid_ask_spread_percent:
            return SlippageResult(False, "Bid-ask spread limit exceeded.", (), max_spread, slippage)
        if slippage > self.max_slippage_percent:
            return SlippageResult(False, "Slippage limit exceeded.", (), max_spread, slippage)
        warnings: list[str] = []
        if max_spread > self.max_bid_ask_spread_percent * 0.8:
            warnings.append("bid_ask_spread_near_limit")
        if slippage > self.max_slippage_percent * 0.8:
            warnings.append("slippage_near_limit")
        return SlippageResult(True, "Slippage guard passed.", tuple(warnings), max_spread, slippage)

    def _max_spread_percent(self, plan: StrategyPlan, context: dict) -> float:
        explicit = context.get("bid_ask_spread_percent")
        if explicit is not None:
            return float(explicit)
        spreads = []
        for leg in plan.legs:
            bid = _optional_float(leg.metadata.get("bid"))
            ask = _optional_float(leg.metadata.get("ask"))
            if bid is None or ask is None:
                continue
            mid = (bid + ask) / 2.0
            spreads.append(safe_divide(ask - bid, mid, 0.0) * 100.0)
        return max(spreads, default=0.0)

    @staticmethod
    def _slippage_percent(plan: StrategyPlan, context: dict, spread_percent: float) -> float:
        explicit = context.get("slippage_percent")
        if explicit is not None:
            return float(explicit)
        explicit_estimate = context.get("slippage_estimate")
        if explicit_estimate is not None:
            return float(explicit_estimate)
        return spread_percent / 2.0 if plan.legs else 0.0


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
