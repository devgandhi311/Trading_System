"""Mandatory deterministic no-look-ahead invariants."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting import BacktestConfig, BacktestEngine, WalkForwardSplitter
from features.feature_pipeline import FeaturePipeline
from ml.dataset_builder import DatasetBuilder
from ml.training_utils import feature_columns
from models import Mode
from scripts.check_lookahead_bias import static_code_audit

pytestmark = [pytest.mark.quantitative, pytest.mark.lookahead]


def _bars(rows: int = 80) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 20_000.0 + index * 3.0 + np.sin(index / 3.0) * 20.0
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=rows, freq="B"),
        "open": close - 4.0, "high": close + 18.0, "low": close - 18.0,
        "close": close, "volume": 1_000_000.0 + index * 100.0,
        "bid": close - 0.5, "ask": close + 0.5,
    })


def test_all_feature_families_are_prefix_invariant(tmp_path: Path) -> None:
    bars, cutoff = _bars(), 44
    pipeline = FeaturePipeline(mode=Mode.BACKTEST, summary_report_path=tmp_path / "summary.xlsx")
    prefix = pipeline.generate_features(bars.iloc[: cutoff + 1], cache=False).features.iloc[-1]
    mutated = bars.copy()
    mutated.loc[cutoff + 1 :, ["open", "high", "low", "close", "volume", "bid", "ask"]] *= 7.0
    historical = pipeline.generate_features(mutated, cache=False).features.iloc[cutoff]
    columns = [column for column in prefix.index if column != "feature_mode"]
    pd.testing.assert_series_equal(prefix[columns], historical[columns], check_names=False, check_dtype=False)


def test_targets_never_enter_ml_feature_matrix() -> None:
    dataset = DatasetBuilder().create_targets(_bars(20))
    columns = feature_columns(dataset, "target_next_day_direction")
    assert columns and not any(column.startswith("target_") for column in columns)


def test_walk_forward_timestamp_boundaries_are_strict() -> None:
    data = _bars(50)
    for window in WalkForwardSplitter(train_size=20, test_size=10).split(data):
        assert window.train(data)["date"].max() < window.test(data)["date"].min()


def test_backtest_fill_is_after_signal_and_future_mutation_is_invariant() -> None:
    bars = _bars(70)
    config = BacktestConfig(rolling_window=5, max_holding_bars=2, use_ml_prediction=False, run_walk_forward=False)
    baseline = BacktestEngine().run(bars.iloc[:45], config=config, save_report=False).trades
    mutated = bars.copy()
    mutated.loc[45:, ["open", "high", "low", "close"]] *= 4.0
    replay = BacktestEngine().run(mutated, config=config, save_report=False).trades
    historical = replay[replay["exit_index"] < 45].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline.reset_index(drop=True), historical, check_dtype=False)
    assert (historical["entry_index"] > historical["signal_index"]).all()
    assert (pd.to_datetime(historical["entry_date"]) > pd.to_datetime(historical["signal_date"])).all()


def test_static_audit_has_no_forbidden_future_patterns() -> None:
    failures = [finding for finding in static_code_audit() if finding.status == "FAIL"]
    assert failures == [], "\n".join(f"{item.file}:{item.line} {item.finding}" for item in failures)


def test_timing_contracts_are_strictly_causal() -> None:
    close = pd.Timestamp("2024-01-02 15:30", tz="Asia/Kolkata")
    execution = pd.Timestamp("2024-01-03 09:15", tz="Asia/Kolkata")
    news = pd.Timestamp("2024-01-02 15:29", tz="Asia/Kolkata")
    assert news <= close < execution
    assert execution >= close  # quote and option snapshot cannot predate their use.
