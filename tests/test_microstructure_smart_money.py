import pandas as pd

from features.microstructure import add_microstructure_features, save_microstructure_features
from features.smart_money import (
    PatternDetection,
    bearish_fair_value_gap,
    bearish_order_block,
    break_of_structure,
    bullish_fair_value_gap,
    bullish_order_block,
    change_of_character,
    displacement_candle,
    equal_highs,
    equal_lows,
    failed_breakout,
    liquidity_sweep_above_previous_high,
    liquidity_sweep_below_previous_low,
)
from market.data_store import DataStore


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-01-01 09:15", "open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000, "bid": 103.9, "ask": 104.1},
            {"date": "2024-01-01 09:16", "open": 104, "high": 106, "low": 103, "close": 105, "volume": 1100, "bid": 104.9, "ask": 105.1},
            {"date": "2024-01-01 09:17", "open": 105, "high": 107, "low": 104, "close": 106, "volume": 4000, "bid": 105.8, "ask": 106.2},
            {"date": "2024-01-01 09:18", "open": 106, "high": 108, "low": 100, "close": 101, "volume": 900, "bid": 100.8, "ask": 101.2},
            {"date": "2024-01-01 09:19", "open": 101, "high": 109, "low": 100, "close": 108, "volume": 1500, "bid": 107.8, "ask": 108.2},
        ]
    )


def test_microstructure_features_are_context_columns(tmp_path) -> None:
    features = add_microstructure_features(_market_frame(), rolling_window=3, opening_range_bars=2)

    expected_columns = {
        "bid_ask_spread_percentage",
        "volume_spike",
        "vwap_deviation",
        "opening_range_high",
        "opening_range_low",
        "opening_range_breakout",
        "previous_day_high_sweep",
        "previous_day_low_sweep",
        "range_expansion",
        "range_compression",
        "liquidity_score",
        "slippage_estimate",
        "abnormal_candle_detection",
    }
    assert expected_columns.issubset(features.columns)
    assert features["liquidity_score"].between(0, 1).all()

    store = DataStore(processed_dir=tmp_path / "processed", features_dir=tmp_path / "features", models_dir=tmp_path / "models")
    _, path = save_microstructure_features(_market_frame(), data_store=store)
    assert path.exists()


def test_smart_money_positive_patterns_return_detection_shape() -> None:
    bullish_fvg = pd.DataFrame(
        [
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 102, "high": 105, "low": 101, "close": 104},
            {"open": 106, "high": 108, "low": 103, "close": 107},
        ]
    )
    bearish_fvg = pd.DataFrame(
        [
            {"open": 110, "high": 112, "low": 109, "close": 111},
            {"open": 108, "high": 109, "low": 105, "close": 106},
            {"open": 104, "high": 107, "low": 101, "close": 102},
        ]
    )
    sweep_above = pd.DataFrame(
        [
            {"open": 100, "high": 105, "low": 99, "close": 101},
            {"open": 101, "high": 106, "low": 100, "close": 102},
            {"open": 102, "high": 108, "low": 101, "close": 105},
        ]
    )
    sweep_below = pd.DataFrame(
        [
            {"open": 100, "high": 105, "low": 99, "close": 101},
            {"open": 101, "high": 103, "low": 98, "close": 100},
            {"open": 100, "high": 102, "low": 96, "close": 99},
        ]
    )
    bullish_ob = pd.DataFrame(
        [
            {"open": 100, "high": 101, "low": 98, "close": 99},
            {"open": 99, "high": 110, "low": 98, "close": 109},
        ]
    )
    bearish_ob = pd.DataFrame(
        [
            {"open": 100, "high": 103, "low": 99, "close": 102},
            {"open": 102, "high": 103, "low": 91, "close": 92},
        ]
    )

    detections = [
        bullish_fair_value_gap(bullish_fvg),
        bearish_fair_value_gap(bearish_fvg),
        liquidity_sweep_above_previous_high(sweep_above),
        liquidity_sweep_below_previous_low(sweep_below),
        bullish_order_block(bullish_ob),
        bearish_order_block(bearish_ob),
    ]

    assert all(isinstance(result, PatternDetection) for result in detections)
    assert all(result.detected for result in detections)
    assert all(0 <= result.strength <= 1 for result in detections)
    assert all(result.explanation for result in detections)


def test_smart_money_structure_patterns_are_bounded() -> None:
    frame = pd.DataFrame(
        [
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 105, "low": 98, "close": 104},
        ]
    )
    failed = pd.DataFrame(
        [
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 103, "low": 99, "close": 100},
        ]
    )

    detections = [
        break_of_structure(frame, lookback=5),
        change_of_character(frame, lookback=3),
        equal_highs(frame.iloc[:5]),
        equal_lows(frame.iloc[:5]),
        failed_breakout(failed),
        displacement_candle(frame),
    ]

    assert all(isinstance(result, PatternDetection) for result in detections)
    assert all(0 <= result.strength <= 1 for result in detections)
    assert all(result.explanation for result in detections)
