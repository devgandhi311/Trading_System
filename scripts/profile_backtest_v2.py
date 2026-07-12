"""Profile the V2 backtester and write debug performance reports."""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import pstats
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.backtest_engine_v2 import BacktestEngineV2, BacktestV2Config
from utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Profile BacktestEngineV2 with cProfile.")
    parser.add_argument("--report-dir", default="reports/backtests")
    parser.add_argument("--debug-dir", default="reports/debug")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--no-stress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run cProfile and generate Excel/SVG reports."""

    args = parse_args()
    setup_logging()
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    engine = BacktestEngineV2(report_dir=args.report_dir, debug_dir=debug_dir)
    max_rows = None if args.all_rows else args.max_rows
    config = BacktestV2Config(max_rows=max_rows, stress_test_enabled=not args.no_stress)
    profiler.enable()
    result = engine.run(config=config)
    profiler.disable()
    stats = pstats.Stats(profiler)
    frame = pd.DataFrame(_stats_rows(stats)).sort_values("cumulative_seconds", ascending=False).reset_index(drop=True)
    module_frame = _module_summary(frame)
    excel_path = debug_dir / "performance_profile.xlsx"
    svg_path = debug_dir / "performance_flamegraph.svg"
    with pd.ExcelWriter(excel_path) as writer:
        frame.head(200).to_excel(writer, sheet_name="Functions", index=False)
        frame.head(20).to_excel(writer, sheet_name="Top 20 Slowest", index=False)
        module_frame.to_excel(writer, sheet_name="Modules", index=False)
        pd.DataFrame([result.summary]).to_excel(writer, sheet_name="Backtest Summary", index=False)
    _write_svg(frame.head(20), svg_path)
    print(f"Performance profile: {excel_path}")
    print(f"Flamegraph: {svg_path}")


def _stats_rows(stats: pstats.Stats) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in stats.stats.items():
        filename, line, function = key
        primitive_calls, total_calls, total_seconds, cumulative_seconds, _callers = value
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "total_seconds": total_seconds,
                "cumulative_seconds": cumulative_seconds,
                "per_call_cumulative": cumulative_seconds / total_calls if total_calls else 0.0,
                "module": _module_name(filename),
            }
        )
    return rows


def _module_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame({"module": [], "cumulative_seconds": []})
    return frame.groupby("module", dropna=False).agg(
        cumulative_seconds=("cumulative_seconds", "sum"),
        total_seconds=("total_seconds", "sum"),
        calls=("total_calls", "sum"),
    ).sort_values("cumulative_seconds", ascending=False).reset_index()


def _module_name(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    parts = normalized.split("/")
    for package in ("backtesting", "strategies", "quant", "features", "regime", "market", "pandas", "numpy", "scipy"):
        if package in parts:
            index = parts.index(package)
            return "/".join(parts[index : min(index + 3, len(parts))])
    return parts[-1]


def _write_svg(frame: pd.DataFrame, path: Path) -> None:
    width = 1200
    row_height = 28
    margin = 20
    height = margin * 2 + max(len(frame), 1) * row_height
    max_value = float(frame["cumulative_seconds"].max()) if not frame.empty else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial, sans-serif;font-size:12px}.title{font-size:16px;font-weight:bold}</style>',
        '<text x="20" y="18" class="title">Backtest V2 cProfile Slowest Functions</text>',
    ]
    for idx, row in frame.reset_index(drop=True).iterrows():
        y = margin + 12 + idx * row_height
        seconds = float(row["cumulative_seconds"])
        bar_width = int((seconds / max(max_value, 1e-9)) * 620)
        label = f"{row['function']} ({row['module']}:{int(row['line'])}) {seconds:.3f}s"
        safe_label = str(label).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f'<rect x="20" y="{y}" width="{bar_width}" height="18" fill="#4f81bd"/>')
        lines.append(f'<text x="{30 + bar_width}" y="{y + 14}">{safe_label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
