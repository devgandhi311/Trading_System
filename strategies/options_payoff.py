"""Option expiry payoff utilities for research and reporting."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Mapping

import numpy as np


def long_call_payoff(expiry_price: float, *, strike: float, premium: float = 0.0, quantity: int = 1) -> float:
    """Return long call payoff at expiry."""

    return (max(float(expiry_price) - float(strike), 0.0) - float(premium)) * int(quantity)


def short_call_payoff(expiry_price: float, *, strike: float, premium: float = 0.0, quantity: int = 1) -> float:
    """Return short call payoff at expiry."""

    return -long_call_payoff(expiry_price, strike=strike, premium=premium, quantity=quantity)


def long_put_payoff(expiry_price: float, *, strike: float, premium: float = 0.0, quantity: int = 1) -> float:
    """Return long put payoff at expiry."""

    return (max(float(strike) - float(expiry_price), 0.0) - float(premium)) * int(quantity)


def short_put_payoff(expiry_price: float, *, strike: float, premium: float = 0.0, quantity: int = 1) -> float:
    """Return short put payoff at expiry."""

    return -long_put_payoff(expiry_price, strike=strike, premium=premium, quantity=quantity)


def bull_call_spread_payoff(
    expiry_price: float,
    *,
    long_strike: float,
    short_strike: float,
    long_premium: float,
    short_premium: float,
    quantity: int = 1,
) -> float:
    """Return bull call debit spread payoff."""

    return long_call_payoff(expiry_price, strike=long_strike, premium=long_premium, quantity=quantity) + short_call_payoff(
        expiry_price,
        strike=short_strike,
        premium=short_premium,
        quantity=quantity,
    )


def bear_put_spread_payoff(
    expiry_price: float,
    *,
    long_strike: float,
    short_strike: float,
    long_premium: float,
    short_premium: float,
    quantity: int = 1,
) -> float:
    """Return bear put debit spread payoff."""

    return long_put_payoff(expiry_price, strike=long_strike, premium=long_premium, quantity=quantity) + short_put_payoff(
        expiry_price,
        strike=short_strike,
        premium=short_premium,
        quantity=quantity,
    )


def bull_put_credit_spread_payoff(
    expiry_price: float,
    *,
    short_strike: float,
    long_strike: float,
    short_premium: float,
    long_premium: float,
    quantity: int = 1,
) -> float:
    """Return bull put credit spread payoff."""

    return short_put_payoff(expiry_price, strike=short_strike, premium=short_premium, quantity=quantity) + long_put_payoff(
        expiry_price,
        strike=long_strike,
        premium=long_premium,
        quantity=quantity,
    )


def bear_call_credit_spread_payoff(
    expiry_price: float,
    *,
    short_strike: float,
    long_strike: float,
    short_premium: float,
    long_premium: float,
    quantity: int = 1,
) -> float:
    """Return bear call credit spread payoff."""

    return short_call_payoff(expiry_price, strike=short_strike, premium=short_premium, quantity=quantity) + long_call_payoff(
        expiry_price,
        strike=long_strike,
        premium=long_premium,
        quantity=quantity,
    )


def iron_condor_payoff(
    expiry_price: float,
    *,
    short_put_strike: float,
    long_put_strike: float,
    short_call_strike: float,
    long_call_strike: float,
    short_put_premium: float,
    long_put_premium: float,
    short_call_premium: float,
    long_call_premium: float,
    quantity: int = 1,
) -> float:
    """Return iron condor payoff."""

    return bull_put_credit_spread_payoff(
        expiry_price,
        short_strike=short_put_strike,
        long_strike=long_put_strike,
        short_premium=short_put_premium,
        long_premium=long_put_premium,
        quantity=quantity,
    ) + bear_call_credit_spread_payoff(
        expiry_price,
        short_strike=short_call_strike,
        long_strike=long_call_strike,
        short_premium=short_call_premium,
        long_premium=long_call_premium,
        quantity=quantity,
    )


def broken_wing_butterfly_payoff(
    expiry_price: float,
    *,
    lower_strike: float,
    body_strike: float,
    upper_strike: float,
    lower_premium: float,
    body_premium: float,
    upper_premium: float,
    option_type: str = "CALL",
    quantity: int = 1,
) -> float:
    """Return 1x-2x-1x broken wing butterfly payoff."""

    option = option_type.upper()
    if option == "PUT":
        payoff = long_put_payoff(expiry_price, strike=lower_strike, premium=lower_premium, quantity=quantity)
        payoff += 2 * short_put_payoff(expiry_price, strike=body_strike, premium=body_premium, quantity=quantity)
        payoff += long_put_payoff(expiry_price, strike=upper_strike, premium=upper_premium, quantity=quantity)
        return payoff
    payoff = long_call_payoff(expiry_price, strike=lower_strike, premium=lower_premium, quantity=quantity)
    payoff += 2 * short_call_payoff(expiry_price, strike=body_strike, premium=body_premium, quantity=quantity)
    payoff += long_call_payoff(expiry_price, strike=upper_strike, premium=upper_premium, quantity=quantity)
    return payoff


def short_strangle_payoff(
    expiry_price: float,
    *,
    call_strike: float,
    put_strike: float,
    call_premium: float,
    put_premium: float,
    quantity: int = 1,
) -> float:
    """Return short strangle payoff."""

    return short_call_payoff(expiry_price, strike=call_strike, premium=call_premium, quantity=quantity) + short_put_payoff(
        expiry_price,
        strike=put_strike,
        premium=put_premium,
        quantity=quantity,
    )


def long_straddle_payoff(
    expiry_price: float,
    *,
    strike: float,
    call_premium: float,
    put_premium: float,
    quantity: int = 1,
) -> float:
    """Return long straddle payoff."""

    return long_call_payoff(expiry_price, strike=strike, premium=call_premium, quantity=quantity) + long_put_payoff(
        expiry_price,
        strike=strike,
        premium=put_premium,
        quantity=quantity,
    )


def calendar_spread_placeholder(expiry_price: float, **kwargs: Any) -> float:
    """Placeholder for calendar spread payoff, which needs term-structure repricing."""

    del expiry_price, kwargs
    raise NotImplementedError("Calendar spread payoff requires pre-expiry volatility and term-structure repricing.")


def calculate_payoff_curve(
    payoff_function: Callable[..., float],
    price_range: Sequence[float],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Return payoff values for a strategy across expiry prices."""

    return [
        {"expiry_price": float(price), "payoff": float(payoff_function(float(price), **kwargs))}
        for price in price_range
    ]


def calculate_max_profit(curve: Iterable[Mapping[str, float]]) -> float:
    """Return maximum payoff from a payoff curve."""

    payoffs = [float(row["payoff"]) for row in curve]
    return max(payoffs) if payoffs else 0.0


def calculate_max_loss(curve: Iterable[Mapping[str, float]]) -> float:
    """Return absolute maximum loss from a payoff curve."""

    payoffs = [float(row["payoff"]) for row in curve]
    return abs(min(payoffs)) if payoffs else 0.0


def calculate_breakevens(curve: Sequence[Mapping[str, float]]) -> list[float]:
    """Estimate breakeven prices where payoff crosses zero."""

    if len(curve) < 2:
        return []
    sorted_curve = sorted(curve, key=lambda row: float(row["expiry_price"]))
    breakevens: list[float] = []
    for left, right in zip(sorted_curve[:-1], sorted_curve[1:]):
        left_price = float(left["expiry_price"])
        right_price = float(right["expiry_price"])
        left_payoff = float(left["payoff"])
        right_payoff = float(right["payoff"])
        if left_payoff == 0:
            breakevens.append(left_price)
            continue
        if left_payoff * right_payoff > 0:
            continue
        denominator = right_payoff - left_payoff
        if denominator == 0:
            continue
        ratio = -left_payoff / denominator
        breakevens.append(left_price + ratio * (right_price - left_price))
    if sorted_curve[-1]["payoff"] == 0:
        breakevens.append(float(sorted_curve[-1]["expiry_price"]))
    return _dedupe_sorted(breakevens)


def payoff_curve_from_legs(price_range: Sequence[float], legs: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Return payoff curve from generic option leg dictionaries."""

    return [
        {"expiry_price": float(price), "payoff": sum(_leg_payoff(float(price), leg) for leg in legs)}
        for price in price_range
    ]


def _leg_payoff(expiry_price: float, leg: Mapping[str, Any]) -> float:
    option_type = str(leg.get("option_type", "CALL")).upper()
    side = str(leg.get("side", "BUY")).upper()
    strike = float(leg["strike"])
    premium = float(leg.get("premium", 0.0))
    quantity = int(leg.get("quantity", 1))
    if option_type in {"CALL", "CE"} and side == "BUY":
        return long_call_payoff(expiry_price, strike=strike, premium=premium, quantity=quantity)
    if option_type in {"CALL", "CE"}:
        return short_call_payoff(expiry_price, strike=strike, premium=premium, quantity=quantity)
    if side == "BUY":
        return long_put_payoff(expiry_price, strike=strike, premium=premium, quantity=quantity)
    return short_put_payoff(expiry_price, strike=strike, premium=premium, quantity=quantity)


def _dedupe_sorted(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or not np.isclose(result[-1], value):
            result.append(float(value))
    return result
