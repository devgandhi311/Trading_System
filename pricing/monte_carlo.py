"""Risk-neutral Monte Carlo option pricing using GBM."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    paths: int = 10_000,
    steps: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM paths with Brownian increments under risk-neutral drift."""

    if spot <= 0 or time_to_expiry <= 0 or volatility <= 0 or paths <= 0 or steps <= 0:
        raise ValueError("spot, time_to_expiry, volatility, paths, and steps must be positive")
    rng = np.random.default_rng(seed)
    dt = time_to_expiry / steps
    brownian_increments = rng.normal(0.0, np.sqrt(dt), size=(paths, steps))
    drift = (risk_free_rate - 0.5 * volatility**2) * dt
    diffusion = volatility * brownian_increments
    log_returns = drift + diffusion
    cumulative = np.cumsum(log_returns, axis=1)
    return spot * np.exp(np.column_stack([np.zeros(paths), cumulative]))


def monte_carlo_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    paths: int = 50_000,
    steps: int = 252,
    seed: int | None = None,
) -> float:
    """Price a European option with risk-neutral GBM Monte Carlo."""

    simulated = simulate_gbm_paths(
        spot,
        time_to_expiry,
        risk_free_rate,
        volatility,
        paths=paths,
        steps=steps,
        seed=seed,
    )
    terminal = simulated[:, -1]
    if option_type.lower() in {"call", "ce"}:
        payoff = np.maximum(terminal - strike, 0.0)
    elif option_type.lower() in {"put", "pe"}:
        payoff = np.maximum(strike - terminal, 0.0)
    else:
        raise ValueError("option_type must be call/put or CE/PE")
    return float(np.exp(-risk_free_rate * time_to_expiry) * payoff.mean())


def scenario_simulation_for_expiry_range(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    paths: int = 10_000,
    steps: int = 252,
    percentiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    seed: int | None = None,
) -> dict[str, float]:
    """Simulate expiry price range scenarios from GBM terminal prices."""

    terminal = simulate_gbm_paths(
        spot,
        time_to_expiry,
        risk_free_rate,
        volatility,
        paths=paths,
        steps=steps,
        seed=seed,
    )[:, -1]
    return {f"p{int(percentile * 100)}": float(np.quantile(terminal, percentile)) for percentile in percentiles}
