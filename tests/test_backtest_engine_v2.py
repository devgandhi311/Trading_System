"""Tests for BacktestEngineV2."""

from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace

from backtesting.backtest_engine_v2 import (
    BacktestEngineV2,
    BacktestV2Config,
    PortfolioState,
    _candidate_costs,
    _prune_expired_positions,
)
from backtesting.transaction_costs import TransactionCostModel
from utils.exceptions import DataValidationError


def test_backtest_engine_v2_runs_and_writes_report(tmp_path) -> None:
    rows = 140
    close = [26000.0 + index * 5.0 for index in range(rows)]
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "open": close,
            "high": [value * 1.005 for value in close],
            "low": [value * 0.995 for value in close],
            "close": close,
            "volume": [100000] * rows,
        }
    )

    result = BacktestEngineV2(report_dir=tmp_path).run(
        historical_data=data,
        config=BacktestV2Config(lookback=40, step=10, max_rows=None, stress_test_enabled=False),
    )

    assert result.report_path.exists()
    assert "trades" in result.summary
    assert set(result.trades.columns) >= {"strategy", "gross_pnl", "net_pnl"} or result.trades.empty
    assert "REAL" in result.report_path.name
    assert "Dataset_Manifest" in pd.ExcelFile(result.report_path).sheet_names
    assert (result.report_path.with_name(f"{result.report_path.stem}_dataset_manifest.json")).exists()


def _candidate(payoff: float = 100.0, legs: int = 1):
    payload = tuple(
        {"side": "BUY", "option_type": "CE", "strike": 100.0, "premium": 5.0, "quantity": 1}
        for _ in range(legs)
    )
    return SimpleNamespace(legs=payload, payoff_curve=({"expiry_price": 110.0, "payoff": payoff},))


def _position(candidate, *, exit_index: int = 2, quantity: int = 1, risk: float = 50.0):
    return {
        "candidate": candidate, "exit_index": exit_index, "expiry_spot": 110.0,
        "quantity": quantity, "risk_amount": risk, "closed": False, "date": pd.Timestamp("2024-01-01"),
    }


def test_expired_positions_close_on_rejected_day_once_and_remove_exposure() -> None:
    state = PortfolioState(1000.0, 1000.0)
    position = _position(_candidate())
    market = pd.Series({"date": pd.Timestamp("2024-01-03"), "close": 110.0})
    first = _prune_expired_positions(2, [position], market, state, TransactionCostModel())
    second = _prune_expired_positions(2, [position], market, state, TransactionCostModel())
    assert len(first.closed_trades) == 1
    assert second.closed_trades == []
    assert first.active_positions == []
    assert state.capital_at_risk == 0.0
    assert state.realized_pnl == pytest.approx(first.closed_trades[0]["net_pnl"])


def test_multiple_expiries_update_equity_before_later_sizing() -> None:
    state = PortfolioState(1000.0, 1000.0)
    market = pd.Series({"date": pd.Timestamp("2024-01-03"), "close": 110.0})
    result = _prune_expired_positions(2, [_position(_candidate(100.0)), _position(_candidate(-50.0))], market, state, TransactionCostModel())
    assert len(result.closed_trades) == 2
    assert state.current_equity == pytest.approx(1000.0 + sum(row["net_pnl"] for row in result.closed_trades))
    assert state.drawdown == pytest.approx(state.current_equity - state.peak_equity)


def test_turnover_costs_are_independent_of_profit_and_scale_with_quantity() -> None:
    model = TransactionCostModel()
    candidate = _candidate(100.0, legs=4)
    one = _candidate_costs(candidate, 110.0, 1, model)
    two = _candidate_costs(candidate, 110.0, 2, model)
    losing = _candidate_costs(_candidate(-100.0, legs=4), 110.0, 1, model)
    assert one["total_turnover"] > 0
    assert one == losing
    assert two["total_turnover"] == pytest.approx(one["total_turnover"] * 2)
    assert two["statutory_charges"] == pytest.approx(one["statutory_charges"] * 2)
    assert one["brokerage"] == 8 * model.brokerage_per_order


def test_sample_data_requires_explicit_configuration(tmp_path) -> None:
    engine = BacktestEngineV2(report_dir=tmp_path)
    data, source, files = engine._load_data(None, BacktestV2Config(use_sample_data=True, sample_rows=55))
    assert source == "SAMPLE"
    assert len(data) == 55
    assert files == ["deterministic_generated_sample"]


def test_real_data_error_is_not_replaced_with_sample(tmp_path) -> None:
    class BrokenLoader:
        def load_30_years_nse_data(self, **kwargs):
            raise DataValidationError("malformed real data")

    engine = BacktestEngineV2(data_loader=BrokenLoader(), report_dir=tmp_path)
    with pytest.raises(DataValidationError, match="malformed real data"):
        engine._load_data(None, BacktestV2Config())
