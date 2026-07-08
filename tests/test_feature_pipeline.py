import pandas as pd

from features.feature_pipeline import FeaturePipeline
from features.regime_features import create_regime_feature_table
from market.data_store import DataStore
from models import Mode


def _feature_frame() -> pd.DataFrame:
    rows = []
    for index in range(30):
        close = 100 + index * 0.5
        rows.append(
            {
                "date": f"2024-01-{index + 1:02d}",
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + index * 10,
                "bid": close - 0.05,
                "ask": close + 0.05,
            }
        )
    return pd.DataFrame(rows)


def test_feature_pipeline_combines_and_caches_features(tmp_path) -> None:
    store = DataStore(
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        models_dir=tmp_path / "models",
    )
    pipeline = FeaturePipeline(
        mode=Mode.BACKTEST,
        data_store=store,
        summary_report_path=tmp_path / "reports" / "feature_summary.xlsx",
    )

    result = pipeline.generate_features(_feature_frame(), dataset_name="combined", rolling_window=5)

    assert result.features_path.exists()
    assert result.regime_features_path.exists()
    assert result.summary_report_path.exists()
    assert "daily_return" in result.features
    assert "rolling_volatility" in result.features
    assert "liquidity_score" in result.features
    assert "smart_money_context_strength" in result.features
    assert "feature_mode" in result.features
    assert "daily_return" in result.regime_features


def test_regime_feature_table_converts_booleans_to_numeric() -> None:
    table = create_regime_feature_table(
        pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "daily_return": 0.01,
                    "opening_range_breakout": True,
                    "bullish_fvg_detected": False,
                    "bullish_fvg_strength": 0.2,
                    "smart_money_context_strength": 0.1,
                }
            ]
        )
    )

    assert table.iloc[0]["opening_range_breakout"] == 1
    assert table.iloc[0]["bullish_fvg_detected"] == 0
