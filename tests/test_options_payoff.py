"""Tests for option payoff utilities."""

from __future__ import annotations

import numpy as np

from strategies.options_payoff import (
    bear_call_credit_spread_payoff,
    bull_call_spread_payoff,
    bull_put_credit_spread_payoff,
    calculate_breakevens,
    calculate_max_loss,
    calculate_max_profit,
    calculate_payoff_curve,
    iron_condor_payoff,
    long_call_payoff,
    long_put_payoff,
    long_straddle_payoff,
    short_call_payoff,
    short_put_payoff,
    short_strangle_payoff,
)


def test_single_option_payoffs() -> None:
    assert long_call_payoff(110.0, strike=100.0, premium=3.0) == 7.0
    assert short_call_payoff(110.0, strike=100.0, premium=3.0) == -7.0
    assert long_put_payoff(90.0, strike=100.0, premium=4.0) == 6.0
    assert short_put_payoff(90.0, strike=100.0, premium=4.0) == -6.0


def test_spread_payoffs() -> None:
    assert bull_call_spread_payoff(
        120.0,
        long_strike=100.0,
        short_strike=110.0,
        long_premium=6.0,
        short_premium=2.0,
    ) == 6.0
    assert bull_put_credit_spread_payoff(
        120.0,
        short_strike=100.0,
        long_strike=90.0,
        short_premium=4.0,
        long_premium=1.0,
    ) == 3.0
    assert bear_call_credit_spread_payoff(
        90.0,
        short_strike=100.0,
        long_strike=110.0,
        short_premium=4.0,
        long_premium=1.0,
    ) == 3.0


def test_iron_condor_and_strategies() -> None:
    condor = iron_condor_payoff(
        100.0,
        short_put_strike=95.0,
        long_put_strike=90.0,
        short_call_strike=105.0,
        long_call_strike=110.0,
        short_put_premium=2.0,
        long_put_premium=0.5,
        short_call_premium=2.0,
        long_call_premium=0.5,
    )
    strangle = short_strangle_payoff(
        100.0,
        call_strike=110.0,
        put_strike=90.0,
        call_premium=2.0,
        put_premium=2.0,
    )
    straddle = long_straddle_payoff(120.0, strike=100.0, call_premium=5.0, put_premium=4.0)

    assert condor == 3.0
    assert strangle == 4.0
    assert straddle == 11.0


def test_payoff_curve_max_loss_profit_and_breakevens() -> None:
    prices = np.linspace(80.0, 120.0, 41)
    curve = calculate_payoff_curve(
        bull_put_credit_spread_payoff,
        prices,
        short_strike=100.0,
        long_strike=90.0,
        short_premium=4.0,
        long_premium=1.0,
    )

    assert calculate_max_profit(curve) == 3.0
    assert calculate_max_loss(curve) == 7.0
    breakevens = calculate_breakevens(curve)
    assert len(breakevens) == 1
    assert np.isclose(breakevens[0], 97.0)
