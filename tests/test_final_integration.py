"""Final integration tests for project entrypoints."""

from __future__ import annotations

from pathlib import Path

import main
import run_backtest
import run_live
import run_paper


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
    assert any(Path(report_dir).glob("*.xlsx"))
