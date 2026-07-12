"""Monte Carlo and scenario stress testing for selected research candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StressTestResult:
    """Stress test output for a strategy candidate."""

    worst_case_loss: float
    expected_loss: float
    probability_of_loss: float
    probability_of_max_loss: float
    scenario_table: tuple[dict[str, Any], ...]
    pass_fail: str
    explanation: str


class MonteCarloStressTester:
    """Stress selected strategy candidates under shocks and expiry distribution samples."""

    def run(
        self,
        *,
        selected_strategy_candidate: Any,
        expiry_distribution: Any,
        volatility_shock_scenarios: Sequence[float] = (0.0, 0.25, 0.50),
        gap_scenarios: Sequence[float] = (-0.03, 0.03),
        news_event_shock: str | None = None,
        max_loss_limit: float | None = None,
    ) -> StressTestResult:
        """Run standard and custom stress scenarios."""

        spot = float(_read_value(expiry_distribution, "expected_expiry_price") or 0.0)
        sigma = float(_read_value(expiry_distribution, "expected_move_points") or max(spot * 0.01, 1.0))
        scenarios = [
            ("normal", spot),
            ("1 sigma down", spot - sigma),
            ("2 sigma down", spot - 2 * sigma),
            ("3 sigma down", spot - 3 * sigma),
            ("1 sigma up", spot + sigma),
            ("2 sigma up", spot + 2 * sigma),
            ("3 sigma up", spot + 3 * sigma),
            ("IV spike", spot),
            ("overnight gap", spot * 0.97),
            ("liquidity shock", spot),
            ("war/crude shock", spot * 0.94),
        ]
        for shock in volatility_shock_scenarios:
            scenarios.append((f"vol shock {shock:.0%}", spot + sigma * float(shock)))
        for gap in gap_scenarios:
            scenarios.append((f"gap {gap:.0%}", spot * (1.0 + float(gap))))
        if news_event_shock:
            scenarios.append((f"news {news_event_shock}", spot * (0.95 if news_event_shock.upper() in {"HIGH", "EXTREME"} else 0.99)))

        table = tuple(
            {
                "scenario": name,
                "expiry_price": price,
                "estimated_pnl": _estimate_pnl(selected_strategy_candidate, price, name),
            }
            for name, price in scenarios
        )
        pnl_values = np.asarray([float(row["estimated_pnl"]) for row in table], dtype=float)
        losses = pnl_values[pnl_values < 0]
        worst_case_loss = abs(float(pnl_values.min())) if pnl_values.size else 0.0
        expected_loss = abs(float(losses.mean())) if losses.size else 0.0
        probability_of_loss = float((pnl_values < 0).mean()) if pnl_values.size else 0.0
        max_loss = float(getattr(selected_strategy_candidate, "max_loss", worst_case_loss) or worst_case_loss)
        probability_of_max_loss = float((pnl_values <= -abs(max_loss) * 0.95).mean()) if pnl_values.size else 0.0
        limit = max_loss_limit if max_loss_limit is not None else max_loss
        passed = worst_case_loss <= abs(float(limit)) and probability_of_max_loss <= 0.20
        return StressTestResult(
            worst_case_loss=worst_case_loss,
            expected_loss=expected_loss,
            probability_of_loss=probability_of_loss,
            probability_of_max_loss=probability_of_max_loss,
            scenario_table=table,
            pass_fail="PASS" if passed else "FAIL",
            explanation=(
                "Stress test passed configured loss tolerance."
                if passed
                else "Stress test failed due to worst-case or max-loss probability."
            ),
        )


def _estimate_pnl(candidate: Any, expiry_price: float, scenario_name: str) -> float:
    curve = list(getattr(candidate, "payoff_curve", ()) or ())
    if curve:
        nearest = min(curve, key=lambda row: abs(float(row["expiry_price"]) - expiry_price))
        pnl = float(nearest["payoff"])
    else:
        pnl = float(getattr(candidate, "expected_value", 0.0) or 0.0)
    max_loss = abs(float(getattr(candidate, "max_loss", 0.0) or 0.0))
    if "IV spike" in scenario_name or "liquidity shock" in scenario_name:
        pnl -= max_loss * 0.10
    if "war/crude" in scenario_name:
        pnl -= max_loss * 0.25
    return max(pnl, -max_loss) if max_loss > 0 else pnl


def _read_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)
