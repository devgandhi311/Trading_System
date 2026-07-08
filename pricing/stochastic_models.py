"""Stochastic model placeholders and shared simulation concepts."""

from __future__ import annotations


def heston_model_placeholder(*args, **kwargs) -> float:
    """Placeholder for Heston stochastic-volatility pricing."""

    raise NotImplementedError("Heston model pricing is not implemented yet")


def jump_diffusion_placeholder(*args, **kwargs) -> float:
    """Placeholder for Merton jump-diffusion pricing."""

    raise NotImplementedError("Jump-diffusion pricing is not implemented yet")
