import numpy as np
import pytest

from pricing import (
    american_option_placeholder,
    call_price,
    crr_binomial_price,
    heston_model_placeholder,
    jump_diffusion_placeholder,
    local_volatility_placeholder,
    monte_carlo_price,
    scenario_simulation_for_expiry_range,
    simulate_gbm_paths,
)


def test_crr_binomial_tree_close_to_black_scholes() -> None:
    bs_price = call_price(100.0, 100.0, 1.0, 0.05, 0.2)
    tree_price = crr_binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, steps=300)

    assert np.isclose(tree_price, bs_price, atol=0.05)


def test_gbm_paths_and_monte_carlo_sanity() -> None:
    paths = simulate_gbm_paths(100.0, 1.0, 0.05, 0.2, paths=1000, steps=10, seed=7)
    mc_price = monte_carlo_price(100.0, 100.0, 1.0, 0.05, 0.2, paths=30_000, steps=60, seed=7)
    bs_price = call_price(100.0, 100.0, 1.0, 0.05, 0.2)

    assert paths.shape == (1000, 11)
    assert np.all(paths[:, 0] == 100.0)
    assert np.isclose(mc_price, bs_price, atol=0.35)


def test_scenario_simulation_percentiles_are_ordered() -> None:
    scenarios = scenario_simulation_for_expiry_range(100.0, 1.0, 0.05, 0.2, seed=11)

    assert scenarios["p5"] < scenarios["p50"] < scenarios["p95"]


def test_placeholder_models_are_explicit() -> None:
    with pytest.raises(NotImplementedError):
        american_option_placeholder()
    with pytest.raises(NotImplementedError):
        heston_model_placeholder()
    with pytest.raises(NotImplementedError):
        jump_diffusion_placeholder()
    with pytest.raises(NotImplementedError):
        local_volatility_placeholder()
