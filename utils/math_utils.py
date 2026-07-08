"""Numeric helpers for trading calculations."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, ROUND_HALF_UP
import math


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers and return default for zero denominators."""

    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a number to an inclusive range."""

    if minimum > maximum:
        raise ValueError("minimum cannot be greater than maximum")
    return max(minimum, min(value, maximum))


def round_to_tick(value: float, tick_size: float) -> float:
    """Round a price to the nearest valid tick size."""

    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    decimal_value = Decimal(str(value))
    decimal_tick = Decimal(str(tick_size))
    ticks = (decimal_value / decimal_tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float(ticks * decimal_tick)


def percent_change(old_value: float, new_value: float, default: float = 0.0) -> float:
    """Return percentage change from old_value to new_value."""

    return safe_divide(new_value - old_value, old_value, default) * 100.0


def percentage_of(value: float, percent: float) -> float:
    """Return a percentage of a value."""

    return value * percent / 100.0


def basis_points_to_percent(basis_points: float) -> float:
    """Convert basis points to percent."""

    return basis_points / 100.0


def mean(values: Iterable[float], default: float = 0.0) -> float:
    """Return arithmetic mean for a non-empty iterable."""

    items = list(values)
    if not items:
        return default
    return sum(items) / len(items)


def is_finite_number(value: float | int) -> bool:
    """Return true for finite int or float values."""

    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: int,
    *,
    side: str,
) -> float:
    """Calculate gross P&L for a BUY or SELL trade."""

    normalized_side = side.upper()
    if normalized_side == "BUY":
        return (exit_price - entry_price) * quantity
    if normalized_side == "SELL":
        return (entry_price - exit_price) * quantity
    raise ValueError(f"Unsupported side: {side}")
