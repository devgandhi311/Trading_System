import numpy as np

from pricing import (
    call_price,
    delta,
    expected_move_from_iv,
    gamma,
    implied_volatility,
    iv_percentile,
    iv_rank,
    probability_itm,
    probability_otm,
    put_price,
    rho,
    theta,
    vega,
)


def test_black_scholes_known_prices_and_greeks() -> None:
    spot = 100.0
    strike = 100.0
    time_to_expiry = 1.0
    risk_free_rate = 0.05
    volatility = 0.20

    assert np.isclose(call_price(spot, strike, time_to_expiry, risk_free_rate, volatility), 10.4506, atol=1e-4)
    assert np.isclose(put_price(spot, strike, time_to_expiry, risk_free_rate, volatility), 5.5735, atol=1e-4)
    assert np.isclose(delta(spot, strike, time_to_expiry, risk_free_rate, volatility), 0.6368, atol=1e-4)
    assert np.isclose(delta(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type="put"), -0.3632, atol=1e-4)
    assert np.isclose(gamma(spot, strike, time_to_expiry, risk_free_rate, volatility), 0.0188, atol=1e-4)
    assert np.isclose(vega(spot, strike, time_to_expiry, risk_free_rate, volatility), 0.3752, atol=1e-4)
    assert np.isclose(theta(spot, strike, time_to_expiry, risk_free_rate, volatility), -0.0176, atol=1e-4)
    assert np.isclose(rho(spot, strike, time_to_expiry, risk_free_rate, volatility), 0.5323, atol=1e-4)


def test_implied_volatility_and_iv_statistics() -> None:
    price = call_price(100.0, 100.0, 1.0, 0.05, 0.20)

    solved = implied_volatility(price, 100.0, 100.0, 1.0, 0.05)

    assert np.isclose(solved, 0.20, atol=1e-5)
    assert iv_rank(0.25, [0.1, 0.2, 0.3]) == 0.75
    assert iv_percentile(0.25, [0.1, 0.2, 0.3]) == 2 / 3


def test_probability_and_expected_move() -> None:
    itm = probability_itm(100.0, 100.0, 1.0, 0.05, 0.20)
    otm = probability_otm(100.0, 100.0, 1.0, 0.05, 0.20)

    assert np.isclose(itm + otm, 1.0)
    assert np.isclose(expected_move_from_iv(100.0, 0.20, 0.25), 10.0)
