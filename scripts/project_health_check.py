"""Run lightweight project health checks and write a Markdown report.

Run:
    python scripts/project_health_check.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.data_loader import DataLoader
from models import Mode
from utils.config_loader import ConfigLoader
from utils.env_loader import get_secret_status, load_project_env
from utils.file_utils import ensure_parent_dir


REPORT_PATH = PROJECT_ROOT / "reports" / "risk_reports" / "project_health_check.md"

REQUIRED_FOLDERS = (
    "config",
    "data/raw/nse_30_years",
    "data/processed",
    "data/features",
    "data/model_data",
    "data/live",
    "data/news_cache",
    "reports/backtests",
    "reports/daily",
    "reports/trade_logs",
    "reports/model_reports",
    "reports/risk_reports",
    "reports/html_reports",
    "logs",
)

DASHBOARD_FILES = (
    "data/live/regime.json",
    "data/live/prediction.json",
    "data/live/risk.json",
    "data/live/kill_switch.json",
    "data/live/news.json",
    "data/live/active_orders.csv",
    "data/live/open_positions.csv",
)


@dataclass(frozen=True)
class CheckResult:
    """One health check result."""

    name: str
    status: str
    detail: str


def main() -> int:
    """Run checks, write report, and print a short summary."""

    checks = [
        _run_check("Config loads", _check_config_loads),
        _run_check(".env loads", _check_env_loads),
        _run_check("News key status", _check_news_keys),
        _run_check("Real data files status", _check_real_data_files),
        _run_check("DataLoader status", _check_data_loader),
        _run_check("Backtest sample status", _check_backtest_sample),
        _run_check("Backtest real-data status", _check_backtest_real),
        _run_check("Paper mode status", _check_paper_mode),
        _run_check("Live remains blocked by default", _check_live_blocked),
        _run_check("Dashboard state files exist", _check_dashboard_files),
        _run_check("Required folders exist", _check_required_folders),
    ]
    report_path = _write_report(checks)
    for result in checks:
        print(f"{result.status}: {result.name} - {result.detail}")
    print(f"Report written: {report_path}")
    return 1 if any(result.status == "FAIL" for result in checks) else 0


def _run_check(name: str, check: Callable[[], tuple[str, str]]) -> CheckResult:
    try:
        status, detail = check()
    except Exception as exc:
        return CheckResult(name, "FAIL", f"{type(exc).__name__}: {exc}")
    return CheckResult(name, status, detail)


def _check_config_loads() -> tuple[str, str]:
    config = ConfigLoader().load()
    return "PASS", f"mode={config.settings.mode.value}, live_trading_enabled={config.settings.live_trading_enabled}"


def _check_env_loads() -> tuple[str, str]:
    loaded = load_project_env(PROJECT_ROOT / ".env")
    return ("PASS" if loaded else "WARNING"), ".env loaded" if loaded else ".env missing or empty"


def _check_news_keys() -> tuple[str, str]:
    keys = ("FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY", "MARKETAUX_API_KEY")
    statuses = {key: get_secret_status(key) for key in keys}
    status = "PASS" if statuses["FINNHUB_API_KEY"] == "YES" else "WARNING"
    return status, ", ".join(f"{key}={value}" for key, value in statuses.items())


def _check_real_data_files() -> tuple[str, str]:
    loader = DataLoader(config_loader=ConfigLoader())
    files = loader._discover_files()
    if not files:
        return "WARNING", f"no supported files found in {loader.raw_dir}"
    return "PASS", f"{len(files)} file(s): {', '.join(path.name for path in files[:5])}"


def _check_data_loader() -> tuple[str, str]:
    result = _run_command(["scripts/check_data_loader.py"], timeout_seconds=45)
    return ("PASS" if result.returncode == 0 else "FAIL"), _compact(result.stdout or result.stderr)


def _check_backtest_sample() -> tuple[str, str]:
    result = _run_command(
        ["run_backtest.py", "--sample", "--rows", "80", "--report-dir", "reports/backtests"],
        timeout_seconds=45,
    )
    return ("PASS" if result.returncode == 0 else "FAIL"), _compact(result.stdout or result.stderr)


def _check_backtest_real() -> tuple[str, str]:
    result = _run_command(
        ["run_backtest.py", "--max-real-rows", "120", "--report-dir", "reports/backtests"],
        timeout_seconds=45,
    )
    return ("PASS" if result.returncode == 0 else "FAIL"), _compact(result.stdout or result.stderr)


def _check_paper_mode() -> tuple[str, str]:
    result = _run_command(["run_paper.py", "--state-dir", "data/live"], timeout_seconds=45)
    return ("PASS" if result.returncode == 0 else "FAIL"), _compact(result.stdout or result.stderr)


def _check_live_blocked() -> tuple[str, str]:
    config = ConfigLoader().load()
    config_blocked = config.settings.mode != Mode.LIVE and not config.settings.live_trading_enabled
    result = _run_command(["run_live.py"], timeout_seconds=30)
    command_blocked = result.returncode != 0 and "LIVE preflight blocked" in (result.stdout + result.stderr)
    if config_blocked and command_blocked:
        return "PASS", "LIVE preflight blocked by default"
    if config_blocked:
        return "WARNING", "config blocks LIVE, but preflight output was unexpected"
    return "FAIL", "LIVE config is enabled"


def _check_dashboard_files() -> tuple[str, str]:
    missing = [path for path in DASHBOARD_FILES if not (PROJECT_ROOT / path).exists()]
    if missing:
        return "WARNING", f"missing: {', '.join(missing)}"
    return "PASS", "all dashboard state files exist"


def _check_required_folders() -> tuple[str, str]:
    missing = [path for path in REQUIRED_FOLDERS if not (PROJECT_ROOT / path).is_dir()]
    if missing:
        return "WARNING", f"missing: {', '.join(missing)}"
    return "PASS", "all required folders exist"


def _run_command(args: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=env,
    )


def _write_report(checks: list[CheckResult]) -> Path:
    path = ensure_parent_dir(REPORT_PATH)
    lines = [
        "# Project Health Check",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for result in checks:
        lines.append(f"| {result.name} | {result.status} | {_escape_markdown(result.detail)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _compact(value: str, *, limit: int = 240) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
