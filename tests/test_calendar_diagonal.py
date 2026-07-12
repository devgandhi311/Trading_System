"""Tests for calendar and diagonal placeholders."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from models import OptionChainSnapshot, OptionContract
from strategies import CalendarSpreadStrategy, DiagonalSpreadStrategy


def _chain(days: int) -> OptionChainSnapshot:
    expiry = date.today() + timedelta(days=days)
    contract = OptionContract("NIFTY26000CE", "NIFTY", expiry, 26000.0, "CE", 75, bid=100.0, ask=102.0, last_price=101.0)
    return OptionChainSnapshot("NIFTY", datetime(2026, 1, 1), 26000.0, expiry, (contract,))


def test_calendar_placeholder_requires_far_chain() -> None:
    result = CalendarSpreadStrategy().analyze(
        near_expiry_option_chain=_chain(7),
        far_expiry_option_chain=None,
        iv_term_structure={"near": 0.2, "far": 0.24},
        expected_move=250.0,
        regime="Sideways low volatility",
    )

    assert result.reject_reason == "Far-expiry option chain is required."
    assert result.legs == ()


def test_diagonal_placeholder_has_correct_interface() -> None:
    result = DiagonalSpreadStrategy().analyze(
        near_expiry_option_chain=_chain(7),
        far_expiry_option_chain=_chain(30),
        iv_term_structure={"near": 0.2, "far": 0.24},
        expected_move=250.0,
        regime="Bullish low volatility",
    )

    assert result.strategy_name == "DiagonalSpread"
    assert "not implemented" in str(result.reject_reason)
