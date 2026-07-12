"""Final integration tests for project entrypoints."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

import main
import run_backtest
import run_live
import run_paper
from market.data_loader import generate_sample_ohlcv


def test_main_safe_init() -> None:
    assert main.main(["init"]) == 0


def test_run_live_blocked_by_default() -> None:
    assert run_live.main([]) == 1


def test_run_paper_smoke_writes_dashboard_state(tmp_path) -> None:
    state_dir = tmp_path / "live_state"

    assert run_paper.main(["--state-dir", str(state_dir)]) == 0
    assert (state_dir / "prediction.json").exists()
    assert (state_dir / "risk.json").exists()
    assert (state_dir / "active_orders.csv").exists()


def test_run_backtest_sample_smoke(tmp_path) -> None:
    report_dir = tmp_path / "backtests"

    assert run_backtest.main(["--sample", "--rows", "45", "--report-dir", str(report_dir), "--dataset-name", "smoke"]) == 0
    reports = list(Path(report_dir).glob("*.xlsx"))
    assert reports
    assert all("SAMPLE" in report.name for report in reports)
    manifest_path = next(Path(report_dir).glob("*SAMPLE*_dataset_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_source"] == "SAMPLE"
    assert manifest["generated_sample"] is True
    assert manifest["source_file_hashes"] == {}
    assert manifest["first_timestamp"]
    assert manifest["last_timestamp"]
    assert manifest["manifest_id"]
    assert manifest["source_type"] == "SAMPLE"
    assert manifest["source_file_sizes"] == {}
    assert manifest["detected_file_formats"] == {}
    assert manifest["normalized_columns"] == ["date", "open", "high", "low", "close", "volume"]
    assert manifest["timezone"] == "Asia/Kolkata"
    assert manifest["loader_version"]
    assert "Dataset_Manifest" in pd.ExcelFile(reports[0]).sheet_names


def test_explicit_sample_generator_is_deterministic() -> None:
    pd.testing.assert_frame_equal(generate_sample_ohlcv(50), generate_sample_ohlcv(50))
    pd.testing.assert_frame_equal(generate_sample_ohlcv(50, seed=17), generate_sample_ohlcv(50, seed=17))


def test_run_backtest_missing_real_data_has_no_sample_fallback(tmp_path, monkeypatch, capsys) -> None:
    raw_dir = tmp_path / "missing"
    original_loader = run_backtest.DataLoader
    monkeypatch.setattr(
        run_backtest,
        "DataLoader",
        lambda **kwargs: original_loader(raw_dir=raw_dir),
    )

    assert run_backtest.main(["--report-dir", str(tmp_path / "reports")]) == 1
    output = capsys.readouterr().out
    assert "Data source: REAL" in output
    assert "No NSE data files found" in output
    assert not list((tmp_path / "reports").glob("*SAMPLE*"))


def test_run_backtest_real_report_and_manifest_identify_source(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = 45
    close = 20_000 + np.arange(rows, dtype=float)
    source = raw_dir / "nifty.csv"
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 1_000_000,
        }
    ).to_csv(source, index=False)
    original_loader = run_backtest.DataLoader
    monkeypatch.setattr(
        run_backtest,
        "DataLoader",
        lambda **kwargs: original_loader(
            raw_dir=raw_dir,
            report_path=tmp_path / "quality.xlsx",
        ),
    )
    report_dir = tmp_path / "reports"

    assert run_backtest.main(["--report-dir", str(report_dir), "--dataset-name", "history"]) == 0
    assert list(report_dir.glob("*REAL*.xlsx"))
    manifest = json.loads(next(report_dir.glob("*REAL*_dataset_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["data_source"] == "REAL"
    assert manifest["generated_sample"] is False
    assert manifest["source_file_hashes"][str(source.resolve())]
    assert manifest["first_timestamp"] <= manifest["last_timestamp"]
    assert manifest["source_file_sizes"][str(source.resolve())] == source.stat().st_size
    assert manifest["detected_file_formats"][str(source.resolve())] == "csv"


def test_real_manifest_matches_exact_sliced_backtest_dataset(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = 80
    close = 20_000 + np.arange(rows, dtype=float)
    pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
        "Open": close - 1, "High": close + 2, "Low": close - 2,
        "Close": close, "Volume": 1_000_000,
    }).to_csv(raw_dir / "nifty.csv", index=False)
    original_loader = run_backtest.DataLoader
    monkeypatch.setattr(run_backtest, "DataLoader", lambda **kwargs: original_loader(raw_dir=raw_dir, report_path=tmp_path / "quality.xlsx"))
    report_dir = tmp_path / "reports"

    assert run_backtest.main(["--report-dir", str(report_dir), "--max-real-rows", "45"]) == 0
    manifest = json.loads(next(report_dir.glob("*REAL*_dataset_manifest.json")).read_text(encoding="utf-8"))
    expected_dates = pd.date_range("2024-01-01", periods=rows, freq="B")[-45:]
    assert manifest["row_count"] == 45
    assert pd.Timestamp(manifest["first_timestamp"]) == expected_dates.min()
    assert pd.Timestamp(manifest["last_timestamp"]) == expected_dates.max()


def test_non_excel_report_path_still_writes_adjacent_manifest(tmp_path) -> None:
    data = generate_sample_ohlcv(40)
    manifest = run_backtest._sample_manifest(data)
    report_path = tmp_path / "report.html"
    report_path.write_text("<html></html>", encoding="utf-8")

    path = run_backtest._write_manifest(manifest, report_path)
    run_backtest._embed_manifest(report_path, manifest)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["generated_sample"] is True
    assert report_path.read_text(encoding="utf-8") == "<html></html>"


def test_run_backtest_insufficient_real_history_fails_closed(tmp_path, monkeypatch, capsys) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = 20
    close = np.arange(rows, dtype=float) + 100
    pd.DataFrame(
        {"Date": pd.date_range("2024-01-01", periods=rows), "Open": close, "High": close + 1,
         "Low": close - 1, "Close": close, "Volume": 1000}
    ).to_csv(raw_dir / "short.csv", index=False)
    original_loader = run_backtest.DataLoader
    monkeypatch.setattr(run_backtest, "DataLoader", lambda **kwargs: original_loader(raw_dir=raw_dir, report_path=tmp_path / "quality.xlsx"))

    assert run_backtest.main(["--report-dir", str(tmp_path / "reports")]) == 1
    assert "Insufficient history" in capsys.readouterr().out
    assert not list((tmp_path / "reports").glob("*.xlsx"))


def test_main_backtest_does_not_force_sample(monkeypatch) -> None:
    received: list[str] = []

    def fake_backtest(argv):
        received.extend(argv)
        return 1

    monkeypatch.setattr(run_backtest, "main", fake_backtest)

    assert main.main(["backtest"]) == 1
    assert received == []


def test_main_backtest_forwards_explicit_sample_arguments(monkeypatch) -> None:
    received: list[str] = []

    def fake_backtest(argv):
        received.extend(argv)
        return 0

    monkeypatch.setattr(run_backtest, "main", fake_backtest)

    assert main.main(["backtest", "--sample", "--rows", "100"]) == 0
    assert received == ["--sample", "--rows", "100"]
