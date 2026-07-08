import pandas as pd

from features.feature_pipeline import FeaturePipeline
from market.data_store import DataStore
from regime import RegimeDetector


def _frame(direction: float = 1.0, *, abnormal: bool = False) -> pd.DataFrame:
    rows = []
    price = 100.0
    for index in range(30):
        price += direction * 0.8
        high = price + 1.0
        low = price - 1.0
        if abnormal and index == 29:
            high = price + 10.0
            low = price - 10.0
        rows.append(
            {
                "date": f"2024-01-{index + 1:02d}",
                "open": price - direction * 0.2,
                "high": high,
                "low": low,
                "close": price,
                "volume": 1000 + index * 5,
                "bid": price - 0.05,
                "ask": price + 0.05,
            }
        )
    return pd.DataFrame(rows)


def test_regime_detector_classifies_bullish_context(tmp_path) -> None:
    store = DataStore(processed_dir=tmp_path / "processed", features_dir=tmp_path / "features", models_dir=tmp_path / "models")
    detector = RegimeDetector(FeaturePipeline(data_store=store, summary_report_path=tmp_path / "feature_summary.xlsx"))

    result = detector.detect(_frame(1.0), symbol="NIFTY")

    assert result.regime_name in {"Bullish low volatility", "Bullish high volatility"}
    assert result.trend_score > 0
    assert 0 <= result.confidence_score <= 1
    assert result.allowed_to_trade
    assert result.explanation


def test_regime_detector_blocks_event_risk(tmp_path) -> None:
    store = DataStore(processed_dir=tmp_path / "processed", features_dir=tmp_path / "features", models_dir=tmp_path / "models")
    detector = RegimeDetector(FeaturePipeline(data_store=store, summary_report_path=tmp_path / "feature_summary.xlsx"))

    result = detector.detect(_frame(1.0, abnormal=True), symbol="NIFTY")

    assert result.regime_name in {"Event-risk no-trade", "Liquidity-trap/fake-breakout"}
    assert not result.allowed_to_trade
