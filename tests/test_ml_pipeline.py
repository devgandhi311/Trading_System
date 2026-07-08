from pathlib import Path

import pandas as pd

from market.data_store import DataStore
from ml.dataset_builder import DatasetBuilder
from ml.model_registry import ModelRegistry
from ml.train_direction_model import train_direction_model
from ml.train_regime_model import train_regime_model
from ml.train_signal_model import train_signal_model
from ml.train_volatility_model import train_volatility_model


def _feature_dataset(rows: int = 80) -> pd.DataFrame:
    data = []
    for index in range(rows):
        close = 100 + ((-1) ** index) * 0.8 + index * 0.1
        data.append(
            {
                "date": f"2024-01-{(index % 28) + 1:02d}",
                "open": close - 0.5,
                "high": close + 1.0 + (index % 3) * 0.2,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + index * 10,
                "daily_return": ((-1) ** index) * 0.01,
                "rolling_volatility": 0.1 + (index % 5) * 0.01,
                "liquidity_score": 0.5 + (index % 3) * 0.1,
                "failed_breakout_detected": index % 7 == 0,
                "regime_name": "bullish" if index % 2 == 0 else "bearish",
            }
        )
    return pd.DataFrame(data)


def test_dataset_builder_creates_targets_splits_and_report(tmp_path: Path) -> None:
    store = DataStore(
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        models_dir=tmp_path / "model_data",
    )
    store.save_features(_feature_dataset(30), "features", prefer_parquet=False)
    builder = DatasetBuilder(
        data_store=store,
        feature_dir=tmp_path / "features",
        model_data_dir=tmp_path / "model_data",
        report_dir=tmp_path / "reports",
    )

    result = builder.build_dataset(dataset_name="supervised", train_window=10, test_window=5)

    assert result.dataset_path.exists()
    assert result.summary_report_path.exists()
    assert result.splits
    assert "target_next_day_direction" in result.dataset
    assert "target_next_day_return" in result.dataset
    assert "target_expected_range" in result.dataset
    assert "target_volatility_expansion" in result.dataset
    assert "target_breakout_failure" in result.dataset
    assert "target_regime_label" in result.dataset


def test_training_modules_save_reports_and_models(tmp_path: Path) -> None:
    dataset = DatasetBuilder().create_targets(_feature_dataset(80))
    registry = ModelRegistry(model_dir=tmp_path / "models", registry_path=tmp_path / "models" / "registry.json")

    direction = train_direction_model(dataset, registry=registry, train_window=30, test_window=10)
    signal = train_signal_model(dataset, registry=registry, train_window=30, test_window=10)
    volatility = train_volatility_model(dataset, registry=registry, train_window=30, test_window=10)
    regime = train_regime_model(dataset, registry=registry, train_window=30, test_window=10)

    assert direction.metrics_path.exists()
    assert direction.registry_entry is not None
    assert signal.metrics_path.exists()
    assert "volatility_expansion" in volatility
    assert "expected_range" in volatility
    assert "hmm" in regime
    assert registry.list_models()
