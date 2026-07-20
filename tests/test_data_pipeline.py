from pathlib import Path
import hashlib

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
    assert report.manifest.data_source == "REAL"
    assert report.manifest.generated_sample is False
    assert report.manifest.row_count == 2
    assert report.manifest.first_timestamp.isoformat().startswith("2024-01-01")
    assert report.manifest.last_timestamp.isoformat().startswith("2024-01-03")
    assert report.manifest.removed_rows == 3
    assert report.manifest.duplicate_rows == 1
    assert report.manifest.invalid_rows == 2
    source_path = str((raw_dir / "nifty.csv").resolve())
    assert report.manifest.source_file_hashes[source_path] == hashlib.sha256((raw_dir / "nifty.csv").read_bytes()).hexdigest()
    assert "Date" in report.manifest.detected_columns
    assert report.manifest.manifest_id
    assert report.manifest.source_type == "REAL"
    assert report.manifest.source_file_sizes[source_path] == (raw_dir / "nifty.csv").stat().st_size
    assert report.manifest.detected_file_formats[source_path] == "csv"
    assert report.manifest.normalized_columns == ("date", "open", "high", "low", "close", "volume")
    assert report.manifest.missing_value_counts["Date"] == 1
    assert report.manifest.timezone == "Asia/Kolkata"
    assert report.manifest.loader_version == DataLoader.LOADER_VERSION
    with pd.ExcelFile(report.report_path) as workbook:
        assert "Dataset_Manifest" in workbook.sheet_names
    discovered = {str(path.resolve()) for path in (raw_dir / "nifty.csv", raw_dir / "nifty_extra.xlsx")}
    assert set(report.manifest.source_file_paths) == discovered
    assert set(report.manifest.source_file_hashes) == discovered
    assert set(report.manifest.source_file_sizes) == discovered
    for path_text in discovered:
        path = Path(path_text)
        assert report.manifest.source_file_hashes[path_text] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert report.manifest.source_file_sizes[path_text] == path.stat().st_size


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


def test_data_loader_missing_real_data_fails_closed(tmp_path: Path) -> None:
    loader = DataLoader(raw_dir=tmp_path / "missing")

    with pytest.raises(DataValidationError, match="No NSE data files found"):
        loader.load_30_years_nse_data()


def test_data_loader_malformed_real_file_fails_closed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "broken.xlsx").write_bytes(b"not an excel workbook")
    loader = DataLoader(raw_dir=raw_dir)

    with pytest.raises(DataValidationError, match="Malformed historical data file"):
        loader.load_30_years_nse_data()


def test_data_loader_invalid_real_schema_fails_closed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame([{"Date": "2024-01-01", "Open": 100.0}]).to_csv(raw_dir / "invalid.csv", index=False)
    loader = DataLoader(raw_dir=raw_dir)

    with pytest.raises(DataValidationError, match="Invalid schema.*Missing required column"):
        loader.load_30_years_nse_data()


def test_data_loader_empty_after_validation_fails_closed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        [{"Date": None, "Open": 100, "High": 110, "Low": 90, "Close": 105, "Volume": 10}]
    ).to_csv(raw_dir / "empty.csv", index=False)

    with pytest.raises(DataValidationError, match="no valid rows"):
        DataLoader(raw_dir=raw_dir).load_30_years_nse_data()


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
