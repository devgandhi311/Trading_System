"""Audit the project for look-ahead bias and future data leakage.

The checks are intentionally conservative: static findings that are safe by
design are marked as INFO/WARNING with rationale, while runtime checks verify
that feature values at time t match a prefix-only calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "risk_reports"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "data",
    "logs",
    "reports",
}
SCAN_DIRS = (
    "backtesting",
    "features",
    "market",
    "ml",
    "news",
    "quant",
    "regime",
    "risk",
    "strategies",
    "execution",
)


@dataclass(frozen=True)
class AuditFinding:
    """One look-ahead audit finding."""

    status: str
    severity: str
    category: str
    file: str
    line: int | str
    code: str
    finding: str
    recommendation: str
    fix_applied: str = ""


def main() -> int:
    """Run static and runtime audits and write Excel/Markdown reports."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    findings: list[AuditFinding] = []
    findings.extend(static_code_audit())
    findings.extend(runtime_causality_checks())
    findings.extend(applied_fix_checks())
    report_path = write_excel_report(findings)
    summary_path = write_markdown_summary(findings, report_path)
    print(f"Look-ahead audit complete: {summary_path}")
    print(f"Excel report: {report_path}")
    fail_count = sum(1 for finding in findings if finding.status == "FAIL")
    warning_count = sum(1 for finding in findings if finding.status == "WARNING")
    print(f"FAIL={fail_count} WARNING={warning_count} PASS={sum(1 for finding in findings if finding.status == 'PASS')}")
    return 1 if fail_count else 0


def static_code_audit() -> list[AuditFinding]:
    """Scan project code for common leakage patterns."""

    findings: list[AuditFinding] = []
    for path in python_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            stripped = line.strip()
            findings.extend(_audit_line(rel, line_no, stripped))
    if not any(item.category == "Centered rolling" for item in findings):
        findings.append(
            AuditFinding(
                "PASS",
                "LOW",
                "Centered rolling",
                "project",
                "-",
                "",
                "No centered rolling windows were found in audited modules.",
                "Keep rolling windows backward-looking only.",
            )
        )
    return findings


def _audit_line(rel: str, line_no: int, line: str) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if "shift(-" in line:
        if rel == "ml/dataset_builder.py" and ("target_" in line or "next_" in line or "failure_source" in line):
            findings.append(
                AuditFinding(
                    "PASS",
                    "INFO",
                    "Negative shift",
                    rel,
                    line_no,
                    line,
                    "Negative shift is used only to create supervised target labels.",
                    "Do not include target_* columns in feature matrices.",
                )
            )
        else:
            findings.append(
                AuditFinding(
                    "FAIL",
                    "HIGH",
                    "Negative shift",
                    rel,
                    line_no,
                    line,
                    "Potential future value leakage through shift(-n).",
                    "Use shift(1+) for features; reserve shift(-1) only for targets excluded from model inputs.",
                )
            )
    if re.search(r"center\s*=\s*True|rolling\([^\n]*center\s*=\s*True", line):
        findings.append(
            AuditFinding(
                "FAIL",
                "HIGH",
                "Centered rolling",
                rel,
                line_no,
                line,
                "Centered rolling windows include future observations.",
                "Use default backward-looking rolling windows.",
            )
        )
    if re.search(r"\.(?:bfill|backfill)\s*\(", line):
        findings.append(
            AuditFinding(
                "FAIL", "HIGH", "Backward fill", rel, line_no, line,
                "Backward filling can copy future observations into historical rows.",
                "Use forward fill only when the data timing contract permits it.",
            )
        )
    if re.search(r"\bfit_transform\(", line):
        findings.append(
            AuditFinding(
                "WARNING",
                "MEDIUM",
                "Global transformer fit",
                rel,
                line_no,
                line,
                "fit_transform can leak if fitted before a time-series split.",
                "Fit transformers inside each walk-forward training fold.",
            )
        )
    if re.search(r"\btrain_test_split\(|(?<!TimeSeries)\bKFold\(|shuffle\s*=\s*True", line):
        findings.append(
            AuditFinding(
                "FAIL",
                "HIGH",
                "Non-temporal split",
                rel,
                line_no,
                line,
                "Random/non-temporal splits can train on future observations.",
                "Use walk-forward or expanding-window splits with chronological order.",
            )
        )
    if "GaussianHMM" in line or "arch_model" in line:
        findings.append(
            AuditFinding(
                "WARNING",
                "MEDIUM",
                "State/vol model fit",
                rel,
                line_no,
                line,
                "HMM/GARCH style models must be fit only on observations available as of prediction time.",
                "Use caller-provided historical windows in backtests; full-history final artifacts are for post-validation use only.",
            )
        )
    if ("date.today()" in line or "now_ist()" in line or "datetime.now(" in line) and rel.startswith(("backtesting/", "strategies/")):
        findings.append(
            AuditFinding(
                "WARNING",
                "MEDIUM",
                "Wall-clock timestamp",
                rel,
                line_no,
                line,
                "Wall-clock dates inside backtest/strategy paths can leak current-time context into historical replay.",
                "Use bar timestamp, snapshot timestamp, or explicit as_of values in backtests.",
            )
        )
    return findings


def runtime_causality_checks() -> list[AuditFinding]:
    """Run focused behavioral checks for causal feature/model plumbing."""

    checks = [
        _check_feature_prefix_equivalence,
        _check_ml_target_exclusion,
        _check_backtest_v2_snapshot_time,
        _check_strategy_selector_snapshot_time,
    ]
    findings: list[AuditFinding] = []
    for check in checks:
        try:
            findings.append(check())
        except Exception as exc:  # pragma: no cover - diagnostic script resilience.
            findings.append(
                AuditFinding(
                    "FAIL",
                    "HIGH",
                    "Runtime causality",
                    check.__name__,
                    "-",
                    "",
                    f"Runtime causality check failed: {type(exc).__name__}: {exc}",
                    "Fix the direct exception and rerun this audit.",
                )
            )
    return findings


def _check_feature_prefix_equivalence() -> AuditFinding:
    from features.feature_pipeline import FeaturePipeline
    from models import Mode

    data = _sample_ohlcv(80)
    pipeline = FeaturePipeline(mode=Mode.BACKTEST)
    full = pipeline.generate_features(data, cache=False).features.reset_index(drop=True)
    mismatches: list[str] = []
    for index in (20, 35, 60):
        prefix = pipeline.generate_features(data.iloc[: index + 1], cache=False).features.reset_index(drop=True)
        full_row = full.iloc[index]
        prefix_row = prefix.iloc[-1]
        for column in _comparable_columns(full):
            left = full_row[column]
            right = prefix_row[column]
            if _values_differ(left, right):
                mismatches.append(f"row={index} column={column}")
                break
    status = "PASS" if not mismatches else "FAIL"
    return AuditFinding(
        status,
        "HIGH" if mismatches else "LOW",
        "Feature causality",
        "features/feature_pipeline.py",
        "-",
        "",
        "Feature values match prefix-only recomputation." if not mismatches else "; ".join(mismatches[:10]),
        "Keep feature engineering causal; investigate mismatched columns if this fails.",
    )


def _check_ml_target_exclusion() -> AuditFinding:
    from ml.training_utils import feature_columns

    dataset = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "daily_return": np.linspace(-0.01, 0.01, 5),
            "target_next_day_return": np.linspace(0.01, -0.01, 5),
            "target_next_day_direction": [1, 0, 1, 0, 1],
        }
    )
    columns = feature_columns(dataset, "target_next_day_direction")
    leaked = [column for column in columns if column.startswith("target_")]
    return AuditFinding(
        "PASS" if not leaked else "FAIL",
        "HIGH" if leaked else "LOW",
        "ML target exclusion",
        "ml/training_utils.py",
        "-",
        "",
        "target_* columns are excluded from training features." if not leaked else f"Leaked targets: {leaked}",
        "Exclude all target_* columns from model feature matrices.",
    )


def _check_backtest_v2_snapshot_time() -> AuditFinding:
    from backtesting.backtest_engine_v2 import _synthetic_option_chain

    as_of = datetime(2020, 1, 15, 15, 30)
    snapshot = _synthetic_option_chain(22000.0, 7, as_of=as_of)
    ok = snapshot.timestamp == as_of and snapshot.expiry.isoformat() == "2020-01-22"
    return AuditFinding(
        "PASS" if ok else "FAIL",
        "HIGH" if not ok else "LOW",
        "Option-chain timestamp",
        "backtesting/backtest_engine_v2.py",
        292,
        "_synthetic_option_chain",
        "Synthetic option-chain snapshots use the historical bar timestamp." if ok else "Synthetic snapshot used wall-clock date/time.",
        "Pass as_of from the replay bar and derive expiry from that date.",
        "Replaced date.today()/now_ist() with replay as_of timestamp.",
    )


def _check_strategy_selector_snapshot_time() -> AuditFinding:
    from datetime import date
    from models import OptionChainSnapshot
    from strategies.strategy_selector import StrategySelector

    timestamp = datetime(2020, 1, 10, 9, 15)
    snapshot = OptionChainSnapshot("NIFTY", timestamp, 22000.0, date(2020, 1, 17), ())
    days = StrategySelector._time_to_expiry_days(snapshot, None)
    return AuditFinding(
        "PASS" if days == 7 else "FAIL",
        "HIGH" if days != 7 else "LOW",
        "Time-to-expiry",
        "strategies/strategy_selector.py",
        220,
        "_time_to_expiry_days",
        "Time-to-expiry uses snapshot timestamp date." if days == 7 else f"Time-to-expiry used non-snapshot date: {days}",
        "Use option-chain snapshot timestamp rather than wall-clock date.",
        "Changed date.today() baseline to snapshot.timestamp.date().",
    )


def applied_fix_checks() -> list[AuditFinding]:
    """Document concrete fixes applied by this audit pass."""

    return [
        AuditFinding(
            "PASS",
            "INFO",
            "Fix applied",
            "backtesting/backtest_engine_v2.py",
            292,
            "_synthetic_option_chain(..., as_of=...)",
            "V2 backtest synthetic option chains now use replay timestamp and historical expiry.",
            "Keep any future backtest market snapshots as-of stamped.",
            "Applied in this audit.",
        ),
        AuditFinding(
            "PASS",
            "INFO",
            "Fix applied",
            "strategies/strategy_selector.py",
            220,
            "_time_to_expiry_days",
            "StrategySelector now computes expiry distance from snapshot timestamp.",
            "Keep strategy selection independent from wall-clock date during replay.",
            "Applied in this audit.",
        ),
    ]


def write_excel_report(findings: list[AuditFinding]) -> Path:
    """Write detailed Excel audit workbook."""

    report_path = REPORT_DIR / "lookahead_bias_report.xlsx"
    frame = pd.DataFrame(asdict(item) for item in findings)
    summary = (
        frame.groupby(["status", "severity", "category"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["status", "severity", "category"])
    )
    with pd.ExcelWriter(report_path) as writer:
        frame.to_excel(writer, sheet_name="Findings", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        frame[frame["fix_applied"].astype(str).str.len() > 0].to_excel(writer, sheet_name="Fixes_Applied", index=False)
    return report_path


def write_markdown_summary(findings: list[AuditFinding], report_path: Path) -> Path:
    """Write human-readable Markdown summary."""

    summary_path = REPORT_DIR / "lookahead_bias_summary.md"
    counts = pd.Series([item.status for item in findings]).value_counts().to_dict()
    failures = [item for item in findings if item.status == "FAIL"]
    warnings = [item for item in findings if item.status == "WARNING"]
    lines = [
        "# Look-Ahead Bias Audit",
        "",
        "## Status",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARNING: {counts.get('WARNING', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        f"- Excel report: `{report_path.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Fixes Applied",
        "- `backtesting/backtest_engine_v2.py`: synthetic option chain now uses the historical replay timestamp.",
        "- `strategies/strategy_selector.py`: time-to-expiry now uses the option-chain snapshot timestamp.",
        "",
        "## High-Risk Failures",
    ]
    if failures:
        lines.extend(
            f"- {item.file}:{item.line} `{item.category}` - {item.finding}"
            for item in failures
        )
    else:
        lines.append("- None found.")
    lines.extend(["", "## Warnings To Review"])
    if warnings:
        lines.extend(
            f"- {item.file}:{item.line} `{item.category}` - {item.finding}"
            for item in warnings[:50]
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "Negative shifts in `ml/dataset_builder.py` are used for supervised labels and are guarded by `ml/training_utils.feature_columns`, which excludes `target_*` columns.",
            "HMM/GARCH warnings mean those model classes require walk-forward caller discipline. The V2 backtest currently passes historical windows ending at the replay bar.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def python_files() -> Iterable[Path]:
    """Yield auditable Python files."""

    for directory in SCAN_DIRS:
        root = PROJECT_ROOT / directory
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            yield path


def _sample_ohlcv(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    close = 22000.0 * np.cumprod(1.0 + rng.normal(0.0001, 0.006, rows))
    open_ = close * (1.0 + rng.normal(0.0, 0.001, rows))
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = rng.integers(100000, 200000, rows)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _comparable_columns(frame: pd.DataFrame) -> list[str]:
    blocked_prefixes = ("feature_mode",)
    columns: list[str] = []
    for column in frame.columns:
        if column == "date" or column.startswith(blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            columns.append(column)
    return columns


def _values_differ(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return False
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return bool(left) != bool(right)
    try:
        return not np.isclose(float(left), float(right), equal_nan=True, rtol=1e-9, atol=1e-9)
    except (TypeError, ValueError):
        return left != right


if __name__ == "__main__":
    sys.exit(main())
