"""Trade count, position, consecutive-loss, and margin exposure limits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExposureResult:
    """Exposure limit result."""

    approved: bool
    reason: str
    warnings: tuple[str, ...]
    margin_usage_percent: float


class ExposureLimits:
    """Validate portfolio exposure limits."""

    def __init__(
        self,
        *,
        max_trades_per_day: int,
        max_open_positions: int,
        max_consecutive_losses: int,
        max_margin_usage_percent: float,
    ) -> None:
        """Create exposure guard."""

        self.max_trades_per_day = int(max_trades_per_day)
        self.max_open_positions = int(max_open_positions)
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.max_margin_usage_percent = float(max_margin_usage_percent)

    def check(
        self,
        *,
        trades_today: int,
        open_positions: int,
        consecutive_losses: int,
        margin_used: float,
        capital: float,
        additional_margin: float = 0.0,
    ) -> ExposureResult:
        """Validate account-level exposure context."""

        if trades_today >= self.max_trades_per_day:
            return ExposureResult(False, "Max trades per day reached.", (), self._margin_usage(margin_used, capital, additional_margin))
        if open_positions >= self.max_open_positions:
            return ExposureResult(False, "Max open positions reached.", (), self._margin_usage(margin_used, capital, additional_margin))
        if consecutive_losses >= self.max_consecutive_losses:
            return ExposureResult(False, "Max consecutive losses reached.", (), self._margin_usage(margin_used, capital, additional_margin))
        margin_usage_percent = self._margin_usage(margin_used, capital, additional_margin)
        if margin_usage_percent > self.max_margin_usage_percent:
            return ExposureResult(False, "Max margin usage exceeded.", (), margin_usage_percent)
        warnings: list[str] = []
        if trades_today >= self.max_trades_per_day * 0.8:
            warnings.append("trades_today_near_limit")
        if open_positions >= self.max_open_positions * 0.8:
            warnings.append("open_positions_near_limit")
        if margin_usage_percent >= self.max_margin_usage_percent * 0.8:
            warnings.append("margin_usage_near_limit")
        return ExposureResult(True, "Exposure limits passed.", tuple(warnings), margin_usage_percent)

    @staticmethod
    def _margin_usage(margin_used: float, capital: float, additional_margin: float) -> float:
        if capital <= 0:
            return 100.0
        return (float(margin_used) + float(additional_margin)) / float(capital) * 100.0
