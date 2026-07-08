import numpy as np
import pandas as pd

from features.returns import add_return_features, save_return_features
from features.volatility import add_volatility_features, save_volatility_features
from market.data_store import DataStore


def _sample_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-01-01", "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 1000},
            {"date": "2024-01-02", "open": 106.0, "high": 112.0, "low": 101.0, "close": 110.0, "volume": 1200},
            {"date": "2024-01-03", "open": 108.0, "high": 111.0, "low": 100.0, "close": 102.0, "volume": 900},
            {"date": "2024-01-04", "open": 103.0, "high": 115.0, "low": 102.0, "close": 114.0, "volume": 1500},
            {"date": "2024-01-05", "open": 113.0, "high": 116.0, "low": 109.0, "close": 111.0, "volume": 1300},
        ]
    )


def test_return_feature_correctness() -> None:
    features = add_return_features(_sample_ohlcv(), rolling_window=2, probability_window=3)

    assert np.isclose(features.loc[1, "daily_return"], 110.0 / 105.0 - 1.0)
    assert np.isclose(features.loc[1, "log_return"], np.log(110.0 / 105.0))
    assert np.isclose(features.loc[1, "overnight_gap"], (106.0 - 105.0) / 105.0)
    assert np.isclose(features.loc[1, "intraday_return"], (110.0 - 106.0) / 106.0)
    assert "gap_up_probability" in features
    assert "average_upside" in features
    assert "average_downside" in features


def test_volatility_feature_correctness() -> None:
    features = add_volatility_features(_sample_ohlcv(), rolling_window=3)

    true_range_first = 110.0 - 95.0
    assert np.isclose(features.loc[0, "atr"], true_range_first)
    assert np.isclose(features.loc[0, "range_percentage"], (110.0 - 95.0) / 105.0)
    assert "rolling_volatility" in features
    assert "realized_volatility" in features
    assert "parkinson_volatility" in features
    assert "garman_klass_volatility" in features
    assert "rolling_skewness" in features
    assert "rolling_kurtosis" in features
    assert "expected_move" in features
    assert "volatility_percentile" in features


def test_feature_saves_to_feature_store(tmp_path) -> None:
    store = DataStore(
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        models_dir=tmp_path / "models",
    )

    return_features, return_path = save_return_features(
        _sample_ohlcv(),
        "returns",
        data_store=store,
    )
    volatility_features, volatility_path = save_volatility_features(
        _sample_ohlcv(),
        "volatility",
        data_store=store,
    )

    assert return_path.exists()
    assert volatility_path.exists()
    assert not return_features.empty
    assert not volatility_features.empty
