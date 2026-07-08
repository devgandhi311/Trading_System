"""CRR binomial tree option pricing."""

from __future__ import annotations

from math import exp, sqrt


def crr_binomial_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    steps: int = 100,
    *,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """Price an option with the Cox-Ross-Rubinstein binomial tree."""

    if american:
        return american_option_placeholder()
    _validate_inputs(spot, strike, time_to_expiry, volatility, steps)
    dt = time_to_expiry / steps
    up = exp(volatility * sqrt(dt))
    down = 1.0 / up
    discount = exp(-risk_free_rate * dt)
    probability = (exp(risk_free_rate * dt) - down) / (up - down)
    if not 0 <= probability <= 1:
        raise ValueError("Invalid CRR probability; check inputs")

    prices = [spot * (up ** (steps - index)) * (down**index) for index in range(steps + 1)]
    values = [_payoff(price, strike, option_type) for price in prices]
    for _ in range(steps):
        values = [
            discount * (probability * values[index] + (1 - probability) * values[index + 1])
            for index in range(len(values) - 1)
        ]
    return float(values[0])


def european_option_price(*args, **kwargs) -> float:
    """Alias for European CRR binomial option pricing."""

    kwargs["american"] = False
    return crr_binomial_price(*args, **kwargs)


def american_option_placeholder() -> float:
    """Placeholder for American exercise logic."""

    raise NotImplementedError("American option exercise logic is not implemented yet")


def _payoff(price: float, strike: float, option_type: str) -> float:
    if option_type.lower() in {"call", "ce"}:
        return max(price - strike, 0.0)
    if option_type.lower() in {"put", "pe"}:
        return max(strike - price, 0.0)
    raise ValueError("option_type must be call/put or CE/PE")


def _validate_inputs(spot: float, strike: float, time_to_expiry: float, volatility: float, steps: int) -> None:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("spot, strike, time_to_expiry, and volatility must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
