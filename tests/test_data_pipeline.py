from pathlib import Path

import pandas as pd
import pytest

from market.data_loader import DataLoader
from market.data_preprocessor import DataPreprocessor
from market.data_store import DataStore
from utils.exceptions import DataValidationError


def test_data_loader_validates_removes_duplicates_and_saves_report(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "nse_30_years"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Date": "2024-01-01", "Open": 100, "High": 110, "Low": 95, "Close": 105, "Volume": 1000},
            {"Date": "2024-01-02", "Open": 105, "High": 104, "Low": 100, "Close": 103, "Volume": 1000},
            {"Date": "2024-01-01", "Open": 101, "High": 111, "Low": 96, "Close": 106, "Volume": 1200},
            {"Date": None, "Open": 101, "High": 111, "Low": 96, "Close": 106, "Volume": 1200},
        ]
    ).to_csv(raw_dir / "nifty.csv", index=False)
    pd.DataFrame(
        [{"Date": "2024-01-03", "Open": 106, "High": 112, "Low": 104, "Close": 110, "Volume": 1300}]
    ).to_excel(raw_dir / "nifty_extra.xlsx", index=False)
    store = DataStore(processed_dir=tmp_path / "processed", features_dir=tmp_path / "features", models_dir=tmp_path / "models")
    loader = DataLoader(
        data_store=store,
        raw_dir=raw_dir,
        report_path=tmp_path / "reports" / "risk_reports" / "data_quality_report.xlsx",
    )

    clean, report = loader.load_30_years_nse_data(preprocess=False)

    assert list(clean["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-03"]
    assert report.removed_bad_ohlc_rows == 1
    assert report.removed_missing_rows == 1
    assert report.removed_duplicate_rows == 1
    assert report.output_path.exists()
    assert report.report_path.exists()


def test_data_loader_auto_detects_alias_columns() -> None:
    loader = DataLoader(raw_dir="missing")
    clean, stats = loader.clean(
        pd.DataFrame(
            [{"Trade Date": "2024-01-01", "O": 1, "H": 2, "L": 0.5, "C": 1.5, "Vol": 10}]
        )
    )

    assert clean.iloc[0]["close"] == 1.5
    assert stats["removed_bad_ohlc_rows"] == 0


def test_data_loader_raises_on_missing_required_columns() -> None:
    loader = DataLoader(raw_dir="missing")

    with pytest.raises(DataValidationError, match="Missing required column"):
        loader.clean(pd.DataFrame([{"Date": "2024-01-01", "Open": 1}]))


def test_data_store_save_load_processed_features_and_models(tmp_path: Path) -> None:
    store = DataStore(processed_dir=tmp_path / "processed", features_dir=tmp_path / "features", models_dir=tmp_path / "models")
    dataframe = pd.DataFrame([{"date": "2024-01-01", "close": 100.0}])

    processed_path = store.save_processed(dataframe, "clean", prefer_parquet=False)
    features_path = store.save_features(dataframe, "features", prefer_parquet=False)
    models_path = store.save_model_dataset(dataframe, "model_data", prefer_parquet=False)

    assert processed_path.exists()
    assert features_path.exists()
    assert models_path.exists()
    assert store.load_processed("clean").iloc[0]["close"] == 100.0
    assert store.load_features("features").iloc[0]["close"] == 100.0
    assert store.load_model_dataset("model_data").iloc[0]["close"] == 100.0


def test_preprocessor_handles_missing_days_anomalies_and_volume() -> None:
    preprocessor = DataPreprocessor(anomaly_return_threshold=0.20)
    raw = pd.DataFrame(
        [
            {"date": "2024-01-01", "open": 100.0, "high": 110.0, "low": 95.0, "close": 100.0, "volume": 1000},
            {"date": "2024-01-03", "open": 210.0, "high": 220.0, "low": 200.0, "close": 210.0, "volume": 2000},
        ]
    )

    clean = preprocessor.preprocess(raw, symbol="NIFTY")

    assert list(clean["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert clean.iloc[1]["volume"] == 0
    assert "volume_normalized" in clean
    assert "volume_zscore" in clean
    assert set(clean["symbol"]) == {"NIFTY"}
